#!/usr/bin/env python3
"""Fit one fixed ModernBERT seed on all official rows and package offline logits.

The three seed choices and all hyperparameters are fixed by E16 development OOF.
This module also provides an offline, hash-checked loader for Kaggle inference.
"""

from __future__ import annotations

import os

# ModernBERT's default compiled path needs Triton on some machines. E15/E16
# were trained in eager mode; set this before importing torch/transformers.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_ID = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SEEDS = (42, 17, 73)
EPOCHS = 12
MAX_LENGTH = 256
BATCH_SIZE = 16
ENCODER_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
LABELS = (
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash",
    "kimi-k3",
)
SMOKE_TEXTS = (
    "Please explain the result in two short sentences.",
    "A small experiment can reveal a surprising pattern.",
    "State the main conclusion and one limitation.",
)
SMOKE_ABSOLUTE_TOLERANCE = 2e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, encoded: dict[str, list[list[int]]], labels: np.ndarray):
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.encoded["input_ids"][index],
            "attention_mask": self.encoded["attention_mask"][index],
            "labels": int(self.labels[index]),
        }


class ModernBertMeanPoolClassifier(nn.Module):
    """Same last-hidden-state mean-pooling head as E15/E16 OOF training."""

    def __init__(self, encoder: nn.Module, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


def score_texts(
    model: ModernBertMeanPoolClassifier,
    tokenizer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return raw 18-class logits in the official label order, without fusion."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not texts:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    target = torch.device(device)
    model.to(target).eval()
    encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            batch = tokenizer.pad(
                [{"input_ids": encoded["input_ids"][i],
                  "attention_mask": encoded["attention_mask"][i]}
                 for i in range(start, stop)],
                padding=True,
                return_tensors="pt",
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=target.type == "cuda"):
                logits = model(batch["input_ids"].to(target), batch["attention_mask"].to(target))
            rows.append(logits.float().cpu().numpy())
    scores = np.concatenate(rows, axis=0)
    if scores.shape != (len(texts), len(LABELS)) or not np.isfinite(scores).all():
        raise RuntimeError("ModernBERT returned non-finite logits or unexpected shape")
    return scores


def artifact_hashes(package: Path) -> dict[str, str]:
    files = [package / "modernbert_state_dict.pt", package / "smoke_reference.json"]
    files.extend(path for path in (package / "tokenizer").rglob("*") if path.is_file())
    files.extend(path for path in (package / "encoder_config").rglob("*") if path.is_file())
    return {str(path.relative_to(package)).replace("\\", "/"): sha256(path)
            for path in sorted(files)}


def load_package(
    path: Path | str,
    device: str | torch.device = "cpu",
) -> tuple[ModernBertMeanPoolClassifier, object, dict]:
    """Load a full-data package offline; verify source, runtime, and artifacts."""
    package = Path(path)
    metadata = read_json(package / "metadata.json")
    if metadata.get("package_schema_version") != 1 or metadata.get("mode") != "full":
        raise ValueError("Expected a full ModernBERT package with schema version 1")
    if metadata.get("model_id") != MODEL_ID or metadata.get("requested_revision") != REVISION:
        raise ValueError("Model ID or pinned revision differs from the E16 recipe")
    if metadata.get("labels") != list(LABELS) or metadata.get("seed") not in SEEDS:
        raise ValueError("Label order or seed differs from the E16 recipe")
    expected_source = metadata.get("source_sha256", {}).get("fit_modernbert_full.py")
    if sha256(Path(__file__)) != expected_source:
        raise ValueError("Inference script differs from the one used for fitting")
    saved_runtime = metadata.get("runtime_versions", {})
    if saved_runtime.get("torch", "").split("+")[0] != torch.__version__.split("+")[0]:
        raise ValueError("PyTorch base version differs from the full-data fit")
    if saved_runtime.get("transformers") != transformers.__version__:
        raise ValueError("Transformers version differs from the full-data fit")
    expected_files = metadata.get("artifact_sha256")
    if not isinstance(expected_files, dict) or artifact_hashes(package) != expected_files:
        raise ValueError("Saved model/tokenizer/config/smoke artifact hashes differ")
    config = AutoConfig.from_pretrained(package / "encoder_config", local_files_only=True)
    if config.model_type != "modernbert" or config.hidden_size != 768:
        raise ValueError("Package encoder config is not ModernBERT-base")
    tokenizer = AutoTokenizer.from_pretrained(
        package / "tokenizer", local_files_only=True, use_fast=True,
    )
    encoder = AutoModel.from_config(config)
    model = ModernBertMeanPoolClassifier(encoder, metadata["training_config"]["dropout"])
    state = torch.load(package / "modernbert_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    return model, tokenizer, metadata


def verify_package(path: Path, device: str | torch.device = "cpu") -> dict:
    model, tokenizer, metadata = load_package(path, device)
    reference = read_json(path / "smoke_reference.json")
    texts = reference.get("texts")
    if texts != list(SMOKE_TEXTS):
        raise ValueError("Smoke texts differ from the fixed synthetic probes")
    expected = np.asarray(reference.get("logits"), dtype=np.float32)
    observed = score_texts(model, tokenizer, texts, batch_size=2, device=device)
    observed_repeat = score_texts(model, tokenizer, texts, batch_size=2, device=device)
    if expected.shape != observed.shape or not np.allclose(
        expected, observed, rtol=0, atol=SMOKE_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError("Reloaded package predictions differ from in-memory fit")
    repeat_max_abs = float(np.max(np.abs(observed - observed_repeat)))
    if repeat_max_abs > 1e-7:
        raise RuntimeError("Reloaded package CPU predictions varied on repeated scoring")
    predicted = np.asarray(LABELS)[observed.argmax(axis=1)]
    if list(predicted) != reference.get("predictions") or not np.array_equal(
        observed.argmax(axis=1), observed_repeat.argmax(axis=1),
    ):
        raise RuntimeError("Reloaded package class predictions differ")
    sorted_scores = np.sort(expected, axis=1)
    return {
        "package_metadata_sha256": sha256(path / "metadata.json"),
        "seed": metadata["seed"],
        "n_smoke_rows": len(texts),
        "max_abs_logit_difference": float(np.max(np.abs(expected - observed))),
        "max_relative_difference_scaled_by_max_1_abs_logit": float(np.max(
            np.abs(expected - observed) / np.maximum(1, np.abs(expected))
        )),
        "absolute_tolerance": SMOKE_ABSOLUTE_TOLERANCE,
        "max_abs_reload_repeat_difference": repeat_max_abs,
        "min_reference_top2_margin": float(np.min(sorted_scores[:, -1] - sorted_scores[:, -2])),
        "all_predictions_equal": True,
        "device": str(device),
    }


def check_e15_config(path: Path, train_hash: str) -> dict:
    config = read_json(path)
    required = {
        "model_id": MODEL_ID, "requested_revision": REVISION,
        "epochs": EPOCHS, "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE, "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE, "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "pooling": "mean",
        "epoch_selection": "fixed final epoch", "train_sha256": train_hash,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"E15 development config changed at {key}")
    return config


def check_snapshot(path: Path) -> dict[str, str]:
    if path.name != REVISION or path.parent.name != "snapshots":
        raise ValueError("Expected the exact pinned Hugging Face snapshot directory")
    required = (
        "config.json", "model.safetensors", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json",
    )
    if any(not (path / name).is_file() for name in required):
        raise ValueError("Pinned ModernBERT snapshot is incomplete")
    return {name: sha256(path / name) for name in required}


def fit(args: argparse.Namespace) -> None:
    from run_e0 import LABELS as OFFICIAL_LABELS, load_and_validate

    if tuple(OFFICIAL_LABELS) != LABELS:
        raise ValueError("Local label order differs from the official E0 definition")
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    if args.seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("Full ModernBERT fit requires CUDA")
    train_hash = sha256(args.train)
    if train_hash != TRAIN_SHA256:
        raise ValueError("Official train.csv SHA-256 differs from the frozen E16 data")
    frame = load_and_validate(args.train)
    if len(frame) != 2699 or frame["prompt_id"].nunique() != 150 or frame["id"].nunique() != 2699:
        raise ValueError("Expected all 2,699 official rows from 150 prompts")
    e15_config = check_e15_config(args.e15_config, train_hash)
    snapshot_hashes = check_snapshot(args.pretrained_snapshot)

    started = time.perf_counter()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, use_fast=True, local_files_only=True,
    )
    encoded_all = tokenizer(
        frame["text"].tolist(), truncation=True, max_length=MAX_LENGTH, padding=False,
    )
    encoded = {key: encoded_all[key] for key in ("input_ids", "attention_mask")}
    numerical_labels = np.asarray([LABELS.index(label) for label in frame["model"]], dtype=np.int64)
    dataset = TextDataset(encoded, numerical_labels)

    def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
        padded = tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
             for row in rows], padding=True, return_tensors="pt",
        )
        padded["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        return padded

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), collate_fn=collate,
        num_workers=0, pin_memory=True,
    )
    encoder = AutoModel.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True,
    )
    actual_revision = getattr(encoder.config, "_commit_hash", None)
    if actual_revision != REVISION or encoder.config.model_type != "modernbert":
        raise RuntimeError("Loaded ModernBERT checkpoint differs from pinned revision")
    if encoder.config.hidden_size != 768:
        raise ValueError("Expected ModernBERT-base hidden size 768")
    model = ModernBertMeanPoolClassifier(encoder).to("cuda:0")
    class_counts = np.bincount(numerical_labels, minlength=len(LABELS))
    if np.any(class_counts == 0):
        raise ValueError("A training class is absent")
    class_weights = torch.tensor(
        len(dataset) / (len(LABELS) * class_counts.astype(np.float32)),
        dtype=torch.float32, device="cuda:0",
    )
    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": ENCODER_LEARNING_RATE},
         {"params": model.classifier.parameters(), "lr": HEAD_LEARNING_RATE}],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = EPOCHS * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses: list[float] = []
    skipped_steps = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            x = batch["input_ids"].to("cuda:0", non_blocking=True)
            mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
            y = batch["labels"].to("cuda:0", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x, mask)
                loss = F.cross_entropy(logits.float(), y, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            else:
                skipped_steps += 1
            loss_sum += float(loss.item()) * len(y)
        losses.append(loss_sum / len(dataset))
        print(f"seed={args.seed} epoch={epoch}/{EPOCHS} train_loss={losses[-1]:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    state_path = args.output_dir / "modernbert_state_dict.pt"
    torch.save(model.cpu().state_dict(), state_path)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    model.encoder.config.save_pretrained(args.output_dir / "encoder_config")
    smoke_scores = score_texts(model, tokenizer, list(SMOKE_TEXTS), batch_size=2, device="cpu")
    smoke_scores_repeat = score_texts(model, tokenizer, list(SMOKE_TEXTS), batch_size=2, device="cpu")
    smoke_repeat_max_abs = float(np.max(np.abs(smoke_scores - smoke_scores_repeat)))
    if smoke_repeat_max_abs > 1e-7 or not np.array_equal(
        smoke_scores.argmax(axis=1), smoke_scores_repeat.argmax(axis=1),
    ):
        raise RuntimeError("In-memory model CPU predictions varied on repeated scoring")
    write_json(args.output_dir / "smoke_reference.json", {
        "texts": list(SMOKE_TEXTS),
        "logits": smoke_scores.tolist(),
        "predictions": [LABELS[index] for index in smoke_scores.argmax(axis=1)],
    })
    training_config = {
        "epochs": EPOCHS,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "pooling": "mean",
        "optimizer": "AdamW",
        "scheduler": "linear warmup/decay, advances only after optimizer update",
        "class_weights": "inverse whole-training class frequency",
        "epoch_selection": "fixed final epoch; no validation selection",
        "mixed_precision": "fp16 autocast with GradScaler",
    }
    metadata = {
        "package_schema_version": 1,
        "mode": "full",
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "pretrained_revision": actual_revision,
        "seed": args.seed,
        "labels": list(LABELS),
        "train_sha256": train_hash,
        "n_training_rows": len(frame),
        "n_training_prompts": int(frame["prompt_id"].nunique()),
        "class_counts": {label: int(class_counts[index]) for index, label in enumerate(LABELS)},
        "training_config": training_config,
        "e15_run_config_sha256": sha256(args.e15_config),
        "e15_fixed_config": {key: e15_config[key] for key in (
            "model_id", "requested_revision", "epochs", "encoder_learning_rate",
            "head_learning_rate", "max_length", "batch_size", "warmup_ratio",
            "weight_decay", "dropout", "pooling", "epoch_selection",
        )},
        "pretrained_snapshot_sha256": snapshot_hashes,
        "artifact_sha256": artifact_hashes(args.output_dir),
        "source_sha256": {
            "fit_modernbert_full.py": sha256(Path(__file__)),
            "run_e0.py": sha256(Path(__file__).with_name("run_e0.py")),
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name("cuda:0"),
        "n_training_steps": total_steps,
        "epoch_train_loss": losses,
        "skipped_optimizer_steps": skipped_steps,
        "fit_seconds": time.perf_counter() - started,
        "smoke_in_memory_repeat_max_abs": smoke_repeat_max_abs,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    smoke = verify_package(args.output_dir, device="cpu")
    write_json(args.output_dir / "smoke_verification.json", smoke)
    print(f"Saved full ModernBERT seed {args.seed}: {args.output_dir}", flush=True)
    print(f"Smoke reload max |logit difference|={smoke['max_abs_logit_difference']:.8g}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit", help="Fit one fixed seed on all official training rows")
    train.add_argument("--train", type=Path, default=Path("data/train.csv"))
    train.add_argument("--e15-config", type=Path,
                       default=Path("outputs/e15_modernbert_base_12ep_eager/run_config.json"))
    train.add_argument("--pretrained-snapshot", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify", help="Reload and compare fixed synthetic logits")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "fit":
            fit(args)
        else:
            report = verify_package(args.package, args.device)
            print(json.dumps(report, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
