#!/usr/bin/env python3
"""E30-A official-only Qwen3-Embedding-8B LoRA, with locked resource and OOF gates.

The lightweight ``preflight`` command requires only the existing tabular packages.
All GPU commands use local files and never download a model. No held-out prompt is
tokenized for training or validation. E30-B and external data are separate studies.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


CONFIG_PATH = Path(__file__).with_name("configs") / "e30_qwen3_embedding8b.json"
SHARDS = tuple(f"model-{number:05d}-of-00004.safetensors" for number in range(1, 5))
REQUIRED_MODEL_FILES = (
    "LICENSE", "config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", *SHARDS,
)
# Official Hugging Face files at the pinned commit, independently matched to
# Qwen's ModelScope mirror. Weight oids are LFS SHA-256.
EXPECTED_SHA256 = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7",
    "model.safetensors.index.json": "ceaf1429bfa18eecc75d079737e8d1fe8dc0627f5ec7e6a985514626d1ac6a77",
    "tokenizer.json": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
    "tokenizer_config.json": "2f58f4bbd7bbce15d683f525954ef3a92cd82f5e06415a9c513859bf8ab72436",
    SHARDS[0]: "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451",
    SHARDS[1]: "dff635b0f6dbbaad2a2d633ef037ec0a39bc165cc1806c712fbd6fcbcb4526c0",
    SHARDS[2]: "30b1d4c53d84eb018f642cad7b373f0aabf79699872d8702c1f38577c0a59a2f",
    SHARDS[3]: "36cbc9c60375693629f25743c1e77ebb1724af58e671b2376463193c7fd21ef6",
}
EXPECTED_E24_METRICS_SHA256 = "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557"
EXPECTED_E24_OOF_SHA256 = "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def config() -> dict:
    value = read_json(CONFIG_PATH)
    if value["experiment"] != "e30_a_qwen3_embedding8b_official_only":
        raise ValueError("E30-A config experiment changed")
    if value["model_repo"] != "Qwen/Qwen3-Embedding-8B":
        raise ValueError("E30-A requires the embedding model, not a generative Qwen variant")
    if value["model_revision"] != "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af":
        raise ValueError("E30-A model revision changed")
    if value["folds"] != list(range(1, N_FOLDS + 1)):
        raise ValueError("E30-A fold list changed")
    if value["deployment_model_count"] != 1:
        raise ValueError("E30-A resource gate assumes one full-data deployment adapter")
    return value


def official_development(train_path: Path, e0_dir: Path, cfg: dict):
    if sha256(train_path) != cfg["official_train_sha256"]:
        raise ValueError("Official train.csv differs from the preregistered SHA-256")
    official = load_and_validate(train_path)
    dev, manifest, baseline, folds, _ = load_e0_split(official, e0_dir, train_path)
    if len(official) != 2699 or official["prompt_id"].nunique() != 150:
        raise ValueError("Official training dataset dimensions changed")
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("E0 development partition dimensions changed")
    if len(manifest) != 150 or manifest["split"].eq("holdout").sum() != 30:
        raise ValueError("E0 locked holdout dimensions changed")
    for fold in cfg["folds"]:
        fit = dev.iloc[np.flatnonzero(folds != fold)]
        valid = dev.iloc[np.flatnonzero(folds == fold)]
        if len(fit) != 1728 or len(valid) != 432:
            raise ValueError(f"Fold {fold} row dimensions changed")
        if fit["prompt_id"].nunique() != 96 or valid["prompt_id"].nunique() != 24:
            raise ValueError(f"Fold {fold} prompt dimensions changed")
        if set(fit["prompt_id"]) & set(valid["prompt_id"]):
            raise ValueError(f"Fold {fold} has prompt leakage")
        if set(fit["model"]) != set(LABELS) or set(valid["model"]) != set(LABELS):
            raise ValueError(f"Fold {fold} is missing a class")
    return official, dev, baseline, folds


def lock_input(args, cfg: dict) -> dict:
    path = args.output_dir / "protocol_lock.json"
    lock = read_json(path)
    expected = {
        "experiment": cfg["experiment"],
        "config_sha256": sha256(CONFIG_PATH),
        "train_sha256": sha256(args.train),
        "e0_manifest_sha256": sha256(args.e0_dir / "split_manifest.csv"),
        "e0_oof_sha256": sha256(args.e0_dir / "oof_predictions.csv"),
        "model_revision": cfg["model_revision"],
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"Protocol lock differs in {key}; do not retune after validation")
    return lock


def preflight(args) -> None:
    cfg = config()
    if (args.output_dir / "protocol_lock.json").exists():
        raise ValueError("Protocol lock already exists; refusing to overwrite it")
    _, dev, _, folds = official_development(args.train, args.e0_dir, cfg)
    report = {
        "experiment": cfg["experiment"],
        "stage": "protocol_lock_before_validation",
        "config_sha256": sha256(CONFIG_PATH),
        "train_sha256": sha256(args.train),
        "e0_manifest_sha256": sha256(args.e0_dir / "split_manifest.csv"),
        "e0_oof_sha256": sha256(args.e0_dir / "oof_predictions.csv"),
        "model_repo": cfg["model_repo"],
        "model_revision": cfg["model_revision"],
        "model_license": cfg["model_license"],
        "development_rows": len(dev),
        "development_prompts": int(dev["prompt_id"].nunique()),
        "fold_rows": {str(fold): int((folds == fold).sum()) for fold in cfg["folds"]},
        "validation_accuracy_computed": False,
        "model_downloaded_by_script": False,
    }
    write_new_json(args.output_dir / "protocol_lock.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def verify_model_dir(model_dir: Path, cfg: dict) -> dict:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete local embedding snapshot; missing {missing}")
    if sum((model_dir / name).stat().st_size for name in SHARDS) < 15_000_000_000:
        raise ValueError("Embedding model weight shards are unexpectedly small")
    index = read_json(model_dir / "model.safetensors.index.json")
    if set(index["weight_map"].values()) != set(SHARDS):
        raise ValueError("Embedding model shard map changed")
    model_config = read_json(model_dir / "config.json")
    for key, value in (("model_type", "qwen3"), ("hidden_size", 4096),
                       ("num_hidden_layers", 36)):
        if model_config.get(key) != value:
            raise ValueError(f"Embedding model config differs in {key}")
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    if tokenizer_config.get("eos_token") != "<|im_end|>" or tokenizer_config.get("pad_token") != "<|endoftext|>":
        raise ValueError("Embedding tokenizer EOS/pad settings changed")
    if "Apache License" not in (model_dir / "LICENSE").read_text(encoding="utf-8"):
        raise ValueError("Expected Apache-2.0 model license is absent")
    observed = {}
    for name in REQUIRED_MODEL_FILES:
        observed[name] = sha256(model_dir / name)
        if name in EXPECTED_SHA256 and observed[name] != EXPECTED_SHA256[name]:
            raise ValueError(f"Official model checksum mismatch for {name}")
    return observed


def runtime_versions(cfg: dict) -> dict:
    wanted = cfg["package_versions"]
    found = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        found[name] = importlib.metadata.version(name)
        required = wanted["torch_prefix"] if name == "torch" else wanted[name]
        if (not found[name].startswith(required) if name == "torch" else found[name] != required):
            raise ValueError(f"Package {name} version {found[name]} differs from {required}")
    return found


def gpu_preflight(cfg: dict) -> None:
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        visible_uuid,
    ):
        raise ValueError("CUDA_VISIBLE_DEVICES must be one physical GPU UUID")

    device_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    devices = [[part.strip() for part in row]
               for row in csv.reader(device_query.stdout.splitlines()) if row]
    matches = [row for row in devices if len(row) == 4 and row[0] == visible_uuid]
    if len(matches) != 1 or "RTX 3090" not in matches[0][1]:
        raise ValueError("The visible UUID must identify one physical RTX 3090")
    try:
        used_mib = int(matches[0][2])
        utilization = int(matches[0][3])
    except ValueError as exc:
        raise ValueError("Cannot verify RTX 3090 memory use and utilization") from exc
    if used_mib > 256 or utilization != 0:
        raise ValueError("RTX 3090 is not idle before E30 model load")

    process_query = subprocess.run(
        ["nvidia-smi", "-i", visible_uuid, "--query-compute-apps=pid",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    for line in process_query.stdout.splitlines():
        pid = line.strip()
        if not pid:
            continue
        if not pid.isdigit():
            raise ValueError("Cannot verify GPU compute-process table")
        raise ValueError("RTX 3090 already has a compute process")

    import torch

    if torch.cuda.device_count() != 1 or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise ValueError("Torch must see exactly one RTX 3090")
    required_free_gib = max(22, cfg["resource_gate"]["max_train_peak_gib"])
    if torch.cuda.mem_get_info(0)[0] < required_free_gib * 2**30:
        raise ValueError("Visible RTX 3090 has less than 22 GiB free")


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_fast=True,
        padding_side="right")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Pinned embedding tokenizer needs pad and EOS tokens")
    sample = tokenizer("E30 tokenizer check", add_special_tokens=True)
    if sample["input_ids"][-1] != tokenizer.eos_token_id:
        raise ValueError("Pinned tokenizer must append EOS before last-token pooling")
    return tokenizer


def build_model(model_dir: Path, cfg: dict, *, adapter_dir: Path | None = None):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        quantization_config=quantization, torch_dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")
    base.config.use_cache = False
    if adapter_dir is None:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        lora = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_targets"], bias="none")
        encoder = get_peft_model(base, lora)
    else:
        encoder = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Linear(4096, len(LABELS), dtype=torch.float32, device="cuda:0")

        def forward(self, input_ids, attention_mask):
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  return_dict=True).last_hidden_state
            positions = attention_mask.long().sum(dim=1) - 1
            rows = torch.arange(states.shape[0], device=states.device)
            pooled = states[rows, positions]
            return self.head(pooled.float())

    return Classifier()


def make_loader(texts: list[str], targets: list[int] | None, tokenizer, cfg: dict,
                *, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, index):
            return texts[index], None if targets is None else targets[index]

    def collate(batch):
        values, target_values = zip(*batch)
        encoded = tokenizer(list(values), padding=True, truncation=True,
                            max_length=cfg["max_length"], add_special_tokens=True,
                            return_tensors="pt")
        if encoded["input_ids"].shape[1] > cfg["max_length"]:
            raise ValueError("Tokenizer exceeded preregistered maximum length")
        if targets is not None:
            encoded["labels"] = torch.tensor(target_values, dtype=torch.long)
        return encoded

    return DataLoader(Rows(), batch_size=cfg["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(cfg["seed"]),
                      num_workers=0, collate_fn=collate)


def train(model, loader, cfg: dict, epochs: int) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or not any(parameter.requires_grad for parameter in model.head.parameters()):
        raise ValueError("No trainable classifier head")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    updates_per_epoch = math.ceil(len(loader) / cfg["gradient_accumulation"])
    total_updates = updates_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(total_updates * cfg["warmup_ratio"]),
        num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    losses, completed_updates, skipped_updates = [], 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to("cuda:0")
            batch = {name: value.to("cuda:0") for name, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**batch)
                loss = F.cross_entropy(logits.float(), labels) / cfg["gradient_accumulation"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite E30 train loss in epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            sum_loss += float(loss.item()) * cfg["gradient_accumulation"] * len(labels)
            if step % cfg["gradient_accumulation"] == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous:
                    scheduler.step()
                    completed_updates += 1
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
            if epochs == 1 and time.perf_counter() - started > cfg["resource_gate"]["max_one_epoch_seconds"]:
                raise TimeoutError("E30 one-epoch resource gate exceeded its time budget")
        losses.append(sum_loss / len(loader.dataset))
        print(f"E30 train epoch {epoch}/{epochs}: loss={losses[-1]:.6f}", flush=True)
    torch.cuda.synchronize(0)
    planned_updates = updates_per_epoch * epochs
    minimum_updates = math.ceil(
        planned_updates * cfg["resource_gate"]["min_completed_optimizer_update_fraction"])
    if completed_updates < minimum_updates:
        raise FloatingPointError(
            f"Only {completed_updates}/{planned_updates} optimizer updates completed; "
            f"minimum is {minimum_updates}")
    return {"epochs": epochs, "train_seconds": time.perf_counter() - started,
            "epoch_loss": losses, "completed_optimizer_updates": completed_updates,
            "skipped_optimizer_updates": skipped_updates,
            "planned_optimizer_updates": planned_updates}


def predict(model, loader) -> np.ndarray:
    import torch

    model.eval()
    parts = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(**{name: value.to("cuda:0") for name, value in batch.items()})
            if logits.shape[1] != len(LABELS) or not torch.isfinite(logits).all():
                raise FloatingPointError("E30 inference produced invalid 18-class logits")
            parts.append(logits.float().cpu().numpy())
    torch.cuda.synchronize(0)
    return np.concatenate(parts)


def save_adapter_head(model, tokenizer, fold_dir: Path) -> None:
    from safetensors.torch import save_file

    adapter_dir = fold_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.encoder.save_pretrained(adapter_dir, safe_serialization=True)
    save_file({name: value.detach().cpu().contiguous()
               for name, value in model.head.state_dict().items()},
              str(fold_dir / "head.safetensors"))
    tokenizer.save_pretrained(fold_dir / "tokenizer")


def reload_model(model_dir: Path, fold_dir: Path, cfg: dict):
    from safetensors.torch import load_file

    adapter_dir = fold_dir / "adapter"
    reloaded = build_model(model_dir, cfg, adapter_dir=adapter_dir)
    reloaded.head.load_state_dict(load_file(str(fold_dir / "head.safetensors")), strict=True)
    return reloaded


def prepare_gpu(args):
    cfg = config()
    lock_input(args, cfg)
    versions = runtime_versions(cfg)
    model_hashes = verify_model_dir(args.model_dir, cfg)
    _, dev, _, folds = official_development(args.train, args.e0_dir, cfg)
    tokenizer = load_tokenizer(args.model_dir)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    gpu_preflight(cfg)
    import torch
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    return cfg, versions, model_hashes, dev, folds, tokenizer


def reset_cuda_peak_stats() -> None:
    import torch

    # Torch 2.6 requires an initialized allocator before resetting peak stats.
    torch.empty(1, device="cuda:0")
    torch.cuda.reset_peak_memory_stats(0)


def resource(args) -> None:
    import torch

    if (args.output_dir / "resource_report.json").exists():
        raise ValueError("Resource report already exists")
    cfg, versions, hashes, dev, folds, tokenizer = prepare_gpu(args)
    train_frame = dev.iloc[np.flatnonzero(folds != 1)]
    target_map = {label: position for position, label in enumerate(LABELS)}
    loader = make_loader(train_frame["text"].tolist(),
                         [target_map[value] for value in train_frame["model"]],
                         tokenizer, cfg, shuffle=True)
    reset_cuda_peak_stats()
    model = build_model(args.model_dir, cfg)
    train_report = train(model, loader, cfg, epochs=1)
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    resource_dir = args.output_dir / "resource_one_epoch"
    resource_dir.mkdir(parents=True, exist_ok=False)
    save_adapter_head(model, tokenizer, resource_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reload_start = time.perf_counter()
    model = reload_model(args.model_dir, resource_dir, cfg)
    reload_seconds = time.perf_counter() - reload_start
    # Fixed unlabeled length/throughput proxy, repeated from official development
    # answers; no hidden test prompt ID, holdout label, or validation accuracy.
    source = train_frame["text"].tolist()
    inference_texts = [source[index % len(source)] for index in range(4000)]
    inference_loader = make_loader(inference_texts, None, tokenizer, cfg, shuffle=False)
    reset_cuda_peak_stats()
    inference_start = time.perf_counter()
    scores = predict(model, inference_loader)
    inference_seconds = time.perf_counter() - inference_start
    inference_peak_allocated = torch.cuda.max_memory_allocated(0)
    inference_peak_reserved = torch.cuda.max_memory_reserved(0)
    if scores.shape != (4000, len(LABELS)):
        raise ValueError("Resource inference row count changed")
    gate_cfg = cfg["resource_gate"]
    gate = {
        "train_peak_under_limit": train_peak_reserved < gate_cfg["max_train_peak_gib"] * 2**30,
        "inference_peak_under_limit": inference_peak_reserved < gate_cfg["max_train_peak_gib"] * 2**30,
        "one_epoch_under_limit": train_report["train_seconds"] <= gate_cfg["max_one_epoch_seconds"],
        "projected_three_epochs_under_limit": train_report["train_seconds"] * cfg["epochs"] <= gate_cfg["max_projected_three_epoch_seconds"],
        "four_thousand_inference_under_limit": inference_seconds <= gate_cfg["max_four_thousand_inference_seconds"],
        "finite_four_thousand_logits": bool(np.isfinite(scores).all()),
    }
    report = {
        "experiment": cfg["experiment"], "stage": "one_fold_resource_only",
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "model_revision": cfg["model_revision"], "model_file_sha256": hashes,
        "packages": versions, "fold_training_side": 1,
        "training_rows": len(train_frame), "training_epochs": 1,
        **train_report,
        "projected_three_epoch_seconds": train_report["train_seconds"] * cfg["epochs"],
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "adapter_reload_seconds": reload_seconds,
        "unlabeled_inference_rows": len(inference_texts),
        "inference_seconds": inference_seconds,
        "inference_peak_allocated_bytes": inference_peak_allocated,
        "inference_peak_reserved_bytes": inference_peak_reserved,
        "projected_five_fold_training_seconds": train_report["train_seconds"] * cfg["epochs"] * len(cfg["folds"]),
        "projected_five_adapter_inference_seconds": (reload_seconds + inference_seconds) * len(cfg["folds"]),
        "deployment_model_count": cfg["deployment_model_count"],
        "five_adapter_kaggle_runtime_proven": False,
        "validation_accuracy_computed": False,
        "hidden_test_runtime_proven": False,
        "gate": gate, "gate_passed": all(gate.values()),
    }
    write_new_json(args.output_dir / "resource_report.json", report)
    print(json.dumps({"gate": gate, "gate_passed": all(gate.values())}, indent=2))


def checked_resource(args, cfg: dict) -> dict:
    report = read_json(args.output_dir / "resource_report.json")
    if (report.get("experiment") != cfg["experiment"] or not report.get("gate_passed")
            or report.get("config_sha256") != sha256(CONFIG_PATH)
            or report.get("protocol_lock_sha256") != sha256(args.output_dir / "protocol_lock.json")
            or report.get("model_revision") != cfg["model_revision"]):
        raise ValueError("E30 resource gate is absent, failed, or stale")
    return report


def fit_fold(args) -> None:
    import torch

    cfg = config()
    lock_input(args, cfg)
    checked_resource(args, cfg)
    if args.fold not in cfg["folds"]:
        raise ValueError("Requested fold is outside the locked E0 folds")
    fold_dir = args.output_dir / f"fold_{args.fold}"
    if fold_dir.exists():
        raise ValueError(f"Fold output already exists: {fold_dir}")
    cfg, versions, hashes, dev, folds, tokenizer = prepare_gpu(args)
    train_idx = np.flatnonzero(folds != args.fold)
    valid_idx = np.flatnonzero(folds == args.fold)
    target_map = {label: position for position, label in enumerate(LABELS)}
    train_frame = dev.iloc[train_idx]
    loader = make_loader(train_frame["text"].tolist(),
                         [target_map[value] for value in train_frame["model"]],
                         tokenizer, cfg, shuffle=True)
    reset_cuda_peak_stats()
    model = build_model(args.model_dir, cfg)
    train_report = train(model, loader, cfg, epochs=cfg["epochs"])
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    fold_dir.mkdir(parents=True, exist_ok=False)
    save_adapter_head(model, tokenizer, fold_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = reload_model(args.model_dir, fold_dir, cfg)
    validation = dev.iloc[valid_idx]
    valid_loader = make_loader(validation["text"].tolist(), None, tokenizer, cfg, shuffle=False)
    reset_cuda_peak_stats()
    inference_start = time.perf_counter()
    scores = predict(model, valid_loader)
    inference_seconds = time.perf_counter() - inference_start
    if scores.shape != (len(valid_idx), len(LABELS)):
        raise ValueError("E30 validation score shape is wrong")
    output = validation.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"}).copy()
    output["fold"] = str(args.fold)
    for index, label in enumerate(LABELS):
        output[f"score_{label}"] = scores[:, index]
    output["pred_model"] = np.asarray(LABELS)[scores.argmax(axis=1)]
    output.to_csv(fold_dir / "oof_predictions.csv", index=False)
    write_new_json(fold_dir / "run.json", {
        "experiment": cfg["experiment"], "fold": args.fold,
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "resource_report_sha256": sha256(args.output_dir / "resource_report.json"),
        "model_revision": cfg["model_revision"], "model_file_sha256": hashes,
        "packages": versions, "train_rows": len(train_idx),
        "validation_rows": len(valid_idx), **train_report,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "validation_inference_seconds": inference_seconds,
        "validation_inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "validation_inference_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "oof_predictions_sha256": sha256(fold_dir / "oof_predictions.csv"),
        "no_epoch_or_threshold_selection": True,
        "holdout_evaluated": False,
    })
    print(f"E30 fold {args.fold} complete; OOF rows={len(output)} (no promotion decision)")


def aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} differs in {key}")


def summarize(args) -> None:
    cfg = config()
    lock_input(args, cfg)
    checked_resource(args, cfg)
    _, dev, _, folds = official_development(args.train, args.e0_dir, cfg)
    pieces = []
    for fold in cfg["folds"]:
        folder = args.output_dir / f"fold_{fold}"
        run = read_json(folder / "run.json")
        path = folder / "oof_predictions.csv"
        for key, expected in (
            ("experiment", cfg["experiment"]), ("fold", fold),
            ("config_sha256", sha256(CONFIG_PATH)),
            ("protocol_lock_sha256", sha256(args.output_dir / "protocol_lock.json")),
            ("resource_report_sha256", sha256(args.output_dir / "resource_report.json")),
            ("oof_predictions_sha256", sha256(path)),
        ):
            if run.get(key) != expected:
                raise ValueError(f"Fold {fold} run metadata differs in {key}")
        frame = load_scores(path)
        if len(frame) != 432 or set(frame["fold"]) != {str(fold)}:
            raise ValueError(f"Fold {fold} row count or identity changed")
        pieces.append(frame)
    e30 = pd.concat(pieces, ignore_index=True).set_index("id", verify_integrity=True)
    if set(e30.index) != set(dev["id"]):
        raise ValueError("Five-fold E30 scores do not cover development rows exactly once")
    e30 = e30.loc[dev["id"]].reset_index()
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if not e30.loc[:, ["id", "prompt_id", "true_model"]].equals(expected):
        raise ValueError("E30 OOF keys differ from the official development rows")
    if not np.array_equal(e30["fold"].to_numpy(dtype=int), folds):
        raise ValueError("E30 OOF fold assignments changed")

    e24_metrics_path = args.e24_oof.with_name("metrics.json")
    if sha256(e24_metrics_path) != EXPECTED_E24_METRICS_SHA256:
        raise ValueError("E24 reference metrics changed from the locked audit")
    if sha256(args.e24_oof) != EXPECTED_E24_OOF_SHA256:
        raise ValueError("Saved E24 reference OOF changed from the locked audit")
    e24_metrics = read_json(e24_metrics_path)
    expected_sources = [
        ("E24 E0c", args.e24_scores, e24_metrics["e24_scores_sha256"]),
        ("RoBERTa", args.roberta_scores, e24_metrics["roberta_scores_sha256"]),
        *[(f"ModernBERT seed {seed}", path,
           e24_metrics["modern_seed_score_sha256"][str(seed)])
          for seed, path in zip((42, 17, 73), args.modern_scores)],
    ]
    for name, path, expected_hash in expected_sources:
        if sha256(path) != expected_hash:
            raise ValueError(f"{name} scores differ from the locked E24 source")
    e24 = load_scores(args.e24_scores)
    roberta = load_scores(args.roberta_scores)
    modern = [load_scores(path) for path in args.modern_scores]
    for name, frame in [("E24", e24), ("RoBERTa", roberta),
                        *[(f"ModernBERT seed {seed}", frame)
                          for seed, frame in zip((42, 17, 73), modern)]]:
        aligned(e30, frame, name)
    true = e30["true_model"].to_numpy()
    baseline_scores = (
        row_standardize(e24.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
        + row_standardize(roberta.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
        + np.mean(np.stack([row_standardize(frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
                            for frame in modern]), axis=0)
    )
    labels = np.asarray(LABELS)
    baseline = labels[baseline_scores.argmax(axis=1)]
    saved_e24 = pd.read_csv(args.e24_oof, dtype=str, keep_default_na=False)
    aligned(e30, saved_e24, "Saved E24 fusion")
    if not np.array_equal(baseline, saved_e24["pred_e16_with_e24"].to_numpy()):
        raise ValueError("Fixed E24 fusion failed exact row-level reproduction")
    e30_scores = e30.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    single = labels[e30_scores.argmax(axis=1)]
    added = labels[(baseline_scores + cfg["promotion_gate"]["fixed_added_branch_weight"]
                    * row_standardize(e30_scores)).argmax(axis=1)]
    scores = {"e24_fixed": float(balanced_accuracy_score(true, baseline)),
              "e30_single": float(balanced_accuracy_score(true, single)),
              "e24_plus_e30_fixed": float(balanced_accuracy_score(true, added))}
    fold_rows = []
    for fold in cfg["folds"]:
        mask = folds == fold
        old = balanced_accuracy_score(true[mask], baseline[mask])
        new = balanced_accuracy_score(true[mask], added[mask])
        fold_rows.append({"fold": fold, "e24_fixed": float(old),
                          "e30_single": float(balanced_accuracy_score(true[mask], single[mask])),
                          "e24_plus_e30_fixed": float(new), "fusion_delta": float(new - old)})
    class_rows = []
    for label in LABELS:
        mask = true == label
        old = float((baseline[mask] == label).mean())
        new = float((added[mask] == label).mean())
        class_rows.append({"model": label, "support": int(mask.sum()),
                           "e24_fixed_recall": old,
                           "e30_single_recall": float((single[mask] == label).mean()),
                           "e24_plus_e30_fixed_recall": new,
                           "fusion_delta": new - old})
    fold_frame = pd.DataFrame(fold_rows)
    class_frame = pd.DataFrame(class_rows)
    gate_cfg = cfg["promotion_gate"]
    delta = scores["e24_plus_e30_fixed"] - scores["e24_fixed"]
    nondecrease = int((fold_frame["fusion_delta"] >= -1e-12).sum())
    worst_class = float(class_frame["fusion_delta"].min())
    gate = {
        "delta_at_least_0_01": delta >= gate_cfg["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": nondecrease >= gate_cfg["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": worst_class > -gate_cfg["max_allowed_class_recall_drop"] + 1e-12,
    }
    cm = confusion_matrix(true, single, labels=LABELS)
    report = {
        "experiment": cfg["experiment"], "stage": "five_fold_oof_analysis",
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "e24_reference_metrics_sha256": sha256(e24_metrics_path),
        "e24_reference_oof_sha256": sha256(args.e24_oof),
        "e24_source_score_sha256": {name: digest for name, _, digest in expected_sources},
        "rows": len(e30), "prompts": int(e30["prompt_id"].nunique()),
        "labels": list(LABELS), "balanced_accuracy": scores,
        "e30_single_confusion_matrix": cm.tolist(),
        "paired_prompt_bootstrap_e24_plus_e30_minus_e24": paired_prompt_bootstrap(
            true, e30["prompt_id"].to_numpy(), baseline, added,
            repeats=gate_cfg["paired_prompt_bootstrap_repeats"], seed=cfg["seed"]),
        "fusion_wrong_to_right": int(((baseline != true) & (added == true)).sum()),
        "fusion_right_to_wrong": int(((baseline == true) & (added != true)).sum()),
        "folds_non_decreasing": nondecrease,
        "worst_class_recall_delta": worst_class,
        "fusion_delta": delta,
        "gate": gate, "gate_passed_against_e24": all(gate.values()),
        "comparison_reference_must_be_rechecked_at_promotion": True,
        "holdout_evaluated": False, "fusion_weight_tuned": False,
    }
    analysis_dir = args.output_dir / "analysis"
    if analysis_dir.exists():
        raise ValueError("E30 analysis output already exists")
    analysis_dir.mkdir(parents=True, exist_ok=False)
    e30.to_csv(analysis_dir / "oof_predictions.csv", index=False)
    fold_frame.to_csv(analysis_dir / "fold_metrics.csv", index=False)
    class_frame.to_csv(analysis_dir / "class_metrics.csv", index=False)
    write_new_json(analysis_dir / "metrics.json", report)
    print(json.dumps({"balanced_accuracy": scores, "gate": gate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("preflight", "resource", "fit-fold", "summarize"):
        stage = stages.add_parser(name)
        stage.add_argument("--train", type=Path, default=Path("data/train.csv"))
        stage.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
        stage.add_argument("--output-dir", type=Path, default=Path("outputs/e30_qwen3_embedding8b_v2"))
        if name in ("resource", "fit-fold"):
            stage.add_argument("--model-dir", type=Path, required=True,
                               help="Local official embedding snapshot; no download is performed")
        if name == "fit-fold":
            stage.add_argument("--fold", type=int, required=True, choices=range(1, 6))
        if name == "summarize":
            stage.add_argument("--e24-scores", type=Path,
                               default=Path("outputs/e24_prvsiyan_fit/decision_scores.csv"))
            stage.add_argument("--roberta-scores", type=Path,
                               default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
            stage.add_argument("--modern-scores", type=Path, nargs=3,
                               default=[Path("outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv"),
                                        Path("outputs/e16_modernbert_seed17_eager/oof_predictions.csv"),
                                        Path("outputs/e16_modernbert_seed73_eager/oof_predictions.csv")])
            stage.add_argument("--e24-oof", type=Path,
                               default=Path("outputs/e24_prvsiyan_analysis/oof_predictions.csv"))
    args = parser.parse_args()
    try:
        {"preflight": preflight, "resource": resource,
         "fit-fold": fit_fold, "summarize": summarize}[args.stage](args)
    except (OSError, ValueError, RuntimeError, ImportError, FloatingPointError,
            TimeoutError) as error:
        print(f"E30 {args.stage} failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
