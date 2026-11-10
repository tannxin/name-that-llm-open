#!/usr/bin/env python3
"""E28 pinned Qwen3-8B QLoRA resource test; never computes validation accuracy."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from run_e0 import LABELS, load_and_validate, sha256
from run_style import load_e0_split


MODEL_REPO = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SHARDS = [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)]
REQUIRED = [
    "LICENSE", "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", *SHARDS,
]
# SHA-256 values verified against the pinned Hugging Face tree and exact
# ModelScope mirror bytes before the resource run. Weight oids are LFS SHA-256.
EXPECTED_SHA256 = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
}
MAX_LENGTH = 256
SEED = 42
FOLD = 1
GRAD_ACCUM = 8
PLANNED_EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.01
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_files(model_dir: Path) -> dict[str, int | None]:
    return {name: (model_dir / name).stat().st_size if (model_dir / name).is_file() else None
            for name in REQUIRED}


def verify_model_dir(model_dir: Path) -> dict[str, int]:
    sizes = model_files(model_dir)
    missing = [name for name, size in sizes.items() if size is None]
    if missing:
        raise ValueError(f"Pinned model snapshot is incomplete; missing: {missing}")
    if any(sizes[name] < 1_000_000_000 for name in SHARDS):
        raise ValueError("Pinned model shard size is unexpectedly small")
    if sum(sizes[name] for name in SHARDS) < 16_000_000_000:
        raise ValueError("Pinned model weight size is unexpectedly small")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapped = set(index["weight_map"].values())
    if mapped != set(SHARDS):
        raise ValueError("Pinned model shard map differs from expected five shards")
    for name in REQUIRED:
        observed = sha256(model_dir / name)
        if observed != EXPECTED_SHA256[name]:
            raise ValueError(f"Pinned model checksum mismatch: {name}: {observed}")
    return {name: int(size) for name, size in sizes.items()}


def preflight(args: argparse.Namespace) -> None:
    import torch

    sizes = model_files(args.model_dir)
    report = {
        "experiment": "e28_qwen3_8b_lora_feasibility",
        "stage": "preflight",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_directory": str(args.model_dir),
        "required_file_bytes": sizes,
        "model_complete": all(size is not None for size in sizes.values()),
        "train_file_sha256": sha256(args.train),
        "train_hash_matches": sha256(args.train) == TRAIN_SHA256,
        "python": sys.version.split()[0],
        "packages": {name: package_version(name) for name in
                     ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        "cuda_visible_device_count": torch.cuda.device_count(),
        "validation_accuracy_computed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def load_train_fold(train_path: Path, e0_dir: Path) -> tuple[list[str], list[int]]:
    if sha256(train_path) != TRAIN_SHA256:
        raise ValueError("Official train.csv SHA differs from the locked protocol")
    frame = load_and_validate(train_path)
    dev, _, _, folds, _ = load_e0_split(frame, e0_dir, train_path)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Development partition differs from the fixed E0/E16 partition")
    train_idx = np.flatnonzero(folds != FOLD)
    if len(train_idx) != 1728 or dev.loc[train_idx, "prompt_id"].nunique() != 96:
        raise ValueError("Fold 1 training side differs from fixed split")
    train = dev.iloc[train_idx]
    if set(train["model"]) != set(LABELS):
        raise ValueError("Fold 1 training side lacks a target class")
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    return train["text"].tolist(), [label_to_id[label] for label in train["model"]]


def make_quantization():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def make_base_model(model_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        quantization_config=make_quantization(),
        torch_dtype=torch.float16,
        device_map={"": 0},
    )


def verify_physical_gpu2_idle() -> str:
    """Recheck the physical card after hashing and before loading a CUDA model."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [tuple(part.strip() for part in line.split(","))
            for line in query.stdout.splitlines()]
    matching = [row for row in rows if len(row) == 4 and row[0] == "2"]
    if len(matching) != 1:
        raise RuntimeError(f"Could not identify physical GPU2: {rows}")
    gpu_uuid = matching[0][1]
    if not gpu_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal the physical GPU2 UUID")
    memory_mib, utilization = int(matching[0][2]), int(matching[0][3])
    if memory_mib > 256 or utilization != 0:
        raise RuntimeError(f"GPU2 is occupied: {memory_mib} MiB, {utilization}%")
    processes = subprocess.run(
        ["nvidia-smi", "-i", gpu_uuid, "--query-compute-apps=pid",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    if re.search(r"\b\d+\b", processes.stdout):
        raise RuntimeError(f"GPU2 has a compute process: {processes.stdout.strip()}")
    return gpu_uuid


def resource(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Resource output directory is not empty: {args.output_dir}")
    sizes = verify_model_dir(args.model_dir)
    versions = {name: package_version(name) for name in
                ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
    if (versions["transformers"] != "4.51.3" or versions["peft"] != "0.15.2"
            or versions["accelerate"] != "1.6.0" or versions["bitsandbytes"] != "0.45.5"):
        raise ValueError(f"Locked package versions differ: {versions}")
    physical_gpu_uuid = verify_physical_gpu2_idle()
    if torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one CUDA device selected by physical GPU2 UUID")
    if "RTX 3090" not in torch.cuda.get_device_name(0):
        raise ValueError("E28 is pinned to one physical RTX 3090 on server GPU2")
    if torch.cuda.mem_get_info(0)[0] < 22 * 2**30:
        raise ValueError("GPU2 is too occupied for the E28 resource test")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    texts, target_ids = load_train_fold(args.train, args.e0_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Pinned Qwen tokenizer has no pad token")

    class Rows(Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, index: int):
            return texts[index], target_ids[index]

    def collate(batch):
        batch_texts, batch_targets = zip(*batch)
        encoded = tokenizer(list(batch_texts), padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt",
                            add_special_tokens=True)
        encoded["labels"] = torch.tensor(batch_targets, dtype=torch.long)
        return encoded

    loader = DataLoader(Rows(), batch_size=1, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED),
                        num_workers=0, collate_fn=collate)
    # Include base-model loading in the measured peak. Torch 2.6 needs the
    # allocator initialized before reset_peak_memory_stats.
    torch.empty(1, device="cuda:0")
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model = make_base_model(args.model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS, modules_to_save=["score"], task_type="SEQ_CLS")
    model = get_peft_model(model, lora)
    if not any("score" in name and param.requires_grad for name, param in model.named_parameters()):
        raise ValueError("The 18-class head is not trainable")
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    total_updates = math.ceil(len(loader) / GRAD_ACCUM) * PLANNED_EPOCHS
    warmup_updates = math.ceil(total_updates * 0.03)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_updates,
        num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda")
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    skipped_updates = 0
    completed_updates = 0
    for step, batch in enumerate(loader, start=1):
        batch = {name: tensor.to("cuda:0") for name, tensor in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(**batch)
            loss = output.loss / GRAD_ACCUM
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite train loss at batch {step}")
        scaler.scale(loss).backward()
        if step % GRAD_ACCUM == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                completed_updates += 1
            else:
                skipped_updates += 1
            optimizer.zero_grad(set_to_none=True)
        if time.perf_counter() - training_started > 3600:
            raise TimeoutError("One-epoch resource gate exceeded 60 minutes")
    train_seconds = time.perf_counter() - training_started
    peak_train_bytes = torch.cuda.max_memory_allocated(0)
    peak_train_reserved_bytes = torch.cuda.max_memory_reserved(0)
    planned_one_epoch_updates = math.ceil(len(loader) / GRAD_ACCUM)
    minimum_completed_updates = math.ceil(planned_one_epoch_updates * 0.95)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "one_epoch_adapter_resource_only"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    # Measure inference peak from the start of adapter reload.
    torch.cuda.reset_peak_memory_stats(0)

    reload_started = time.perf_counter()
    model = make_base_model(args.model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    reload_seconds = time.perf_counter() - reload_started
    # Repetition tests batch/sequence throughput only; no label or accuracy is read.
    inference_texts = [texts[index % len(texts)] for index in range(4000)]
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for index, sample in enumerate(inference_texts, start=1):
            encoded = tokenizer(sample, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt", add_special_tokens=True)
            encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
            logits = model(**encoded).logits
            if logits.shape != (1, len(LABELS)) or not torch.isfinite(logits).all():
                raise FloatingPointError(f"Invalid 18-class logits at inference row {index}")
    torch.cuda.synchronize(0)
    inference_seconds = time.perf_counter() - inference_started
    peak_inference_bytes = torch.cuda.max_memory_allocated(0)
    peak_inference_reserved_bytes = torch.cuda.max_memory_reserved(0)
    projected_training = train_seconds * PLANNED_EPOCHS
    gate = {
        "train_peak_under_22_gib": peak_train_reserved_bytes < 22 * 2**30,
        "inference_peak_under_22_gib": peak_inference_reserved_bytes < 22 * 2**30,
        "optimizer_updates_at_least_95pct": completed_updates >= minimum_completed_updates,
        "one_epoch_under_60_minutes": train_seconds <= 3600,
        "projected_three_epochs_under_3_hours": projected_training <= 3 * 3600,
        "four_thousand_inference_under_3_hours": inference_seconds <= 3 * 3600,
        "no_nonfinite_or_oom": True,
    }
    report = {
        "experiment": "e28_qwen3_8b_lora_feasibility",
        "stage": "one_fold_resource_only",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_files_bytes": sizes,
        "model_sha256_verified": True,
        "physical_gpu_uuid": physical_gpu_uuid,
        "train_sha256": sha256(args.train),
        "fold": FOLD,
        "training_rows": len(texts),
        "training_epochs_completed": 1,
        "candidate_training_epochs": PLANNED_EPOCHS,
        "training_seconds": train_seconds,
        "projected_three_epoch_seconds": projected_training,
        "train_peak_allocated_bytes": peak_train_bytes,
        "train_peak_reserved_bytes": peak_train_reserved_bytes,
        "adapter_reload_seconds": reload_seconds,
        "inference_rows": len(inference_texts),
        "inference_seconds": inference_seconds,
        "inference_peak_allocated_bytes": peak_inference_bytes,
        "inference_peak_reserved_bytes": peak_inference_reserved_bytes,
        "completed_optimizer_updates": completed_updates,
        "skipped_optimizer_updates": skipped_updates,
        "planned_one_epoch_optimizer_updates": planned_one_epoch_updates,
        "minimum_completed_optimizer_updates": minimum_completed_updates,
        "packages": versions,
        "total_wall_seconds": time.perf_counter() - started,
        "validation_accuracy_computed": False,
        "validation_scores_saved": False,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "kaggle_hardware_equivalence_proven": False,
    }
    (args.output_dir / "resource_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "resource"):
        sub = stages.add_parser(stage)
        sub.add_argument("--model-dir", type=Path,
                         default=Path("data/models/qwen3-8b-b968826d"))
        sub.add_argument("--train", type=Path, default=Path("data/train.csv"))
        sub.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
        sub.add_argument("--output-dir", type=Path,
                         default=Path(f"outputs/e28_qwen8b_{stage}"))
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight(args)
    else:
        resource(args)


if __name__ == "__main__":
    main()
