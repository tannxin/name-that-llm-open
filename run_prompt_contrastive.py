#!/usr/bin/env python3
"""E4: fixed prompt-pair CE and two contrastive negative-set ablations.

Each training batch contains all 18 model responses from each of two prompts.
The other prompt supplies one positive per label. Contrastive conditions either
use only cross-prompt negatives or add the 17 same-prompt hard negatives.
E0's locked 30-prompt holdout is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from run_e0 import LABELS, load_and_validate


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CONDITIONS = (
    ("ce_only", 0.0, True),
    ("ce_supcon_crossprompt", 0.2, False),
    ("ce_supcon", 0.2, True),
)
N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("outputs/e0_tfidf_svm/split_manifest.csv"),
    )
    parser.add_argument(
        "--e0-oof", type=Path,
        default=Path("outputs/e0_tfidf_svm/oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/e4_prompt_contrastive"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument(
        "--only-condition", choices=[name for name, _, _ in CONDITIONS],
        help="Run one condition in a new output directory (for an interrupted or added ablation)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.splitlines()[0].strip() or None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_development(
    train_path: Path, manifest_path: Path, e0_oof_path: Path
) -> pd.DataFrame:
    frame = load_and_validate(train_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("E0 split manifest must have prompt_id,split columns")
    if manifest["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt in E0 split manifest")
    if set(manifest["prompt_id"]) != set(frame["prompt_id"]):
        raise ValueError("E0 split manifest prompts differ from train.csv")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("Unexpected split name in E0 manifest")
    if int(manifest["split"].eq("holdout").sum()) != 30:
        raise ValueError("Expected 30 locked holdout prompts")

    dev_prompts = set(manifest.loc[manifest["split"] == "development", "prompt_id"])
    development = frame.loc[frame["prompt_id"].isin(dev_prompts)].copy()
    if len(dev_prompts) != 120 or len(development) != 2160:
        raise ValueError("Expected 120 complete development prompts and 2160 rows")
    for prompt_id, group in development.groupby("prompt_id"):
        if len(group) != len(LABELS) or set(group["model"]) != set(LABELS):
            raise ValueError(f"Development prompt {prompt_id} lacks a full 18-model set")

    oof = pd.read_csv(e0_oof_path, dtype=str, keep_default_na=False)
    required = {"id", "prompt_id", "true_model", "fold"}
    if not required.issubset(oof.columns) or oof["id"].duplicated().any():
        raise ValueError("E0 OOF assignments are missing required fields or duplicate IDs")
    if set(oof["id"]) != set(development["id"]):
        raise ValueError("E0 OOF IDs differ from development IDs")
    development = development.merge(
        oof.loc[:, ["id", "prompt_id", "true_model", "fold"]],
        on="id", validate="one_to_one", suffixes=("", "_e0"),
    )
    if not development["prompt_id"].equals(development["prompt_id_e0"]):
        raise ValueError("E0 OOF prompt assignments differ from train.csv")
    if not development["model"].equals(development["true_model"]):
        raise ValueError("E0 OOF labels differ from train.csv")
    if set(development["fold"]) != {str(i) for i in range(1, N_FOLDS + 1)}:
        raise ValueError("E0 OOF must have exactly folds 1..5")
    if development.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("A prompt occurs in more than one E0 fold")
    return development.reset_index(drop=True)


def make_pair_schedule(dev: pd.DataFrame, seed: int, epochs: int) -> dict[int, list[list[tuple[str, str]]]]:
    """Return 5 folds × epochs × batches, independent of model stochasticity."""
    schedules: dict[int, list[list[tuple[str, str]]]] = {}
    for fold in range(1, N_FOLDS + 1):
        train_prompts = np.array(sorted(dev.loc[dev["fold"] != str(fold), "prompt_id"].unique()))
        valid_prompts = set(dev.loc[dev["fold"] == str(fold), "prompt_id"])
        if len(train_prompts) != 96 or len(valid_prompts) != 24:
            raise ValueError(f"Fold {fold} lacks 96/24 train/validation prompts")
        if set(train_prompts) & valid_prompts:
            raise ValueError(f"Fold {fold} leaks prompt groups")
        rng = np.random.default_rng(seed + fold)
        fold_epochs: list[list[tuple[str, str]]] = []
        for _ in range(epochs):
            shuffled = rng.permutation(train_prompts)
            fold_epochs.append([
                (str(shuffled[i]), str(shuffled[i + 1]))
                for i in range(0, len(shuffled), 2)
            ])
        schedules[fold] = fold_epochs
    return schedules


class PromptClassifier(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID, revision=REVISION)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        logits = self.classifier(self.dropout(pooled))
        normalized = F.normalize(pooled.float(), p=2, dim=1)
        return logits, normalized


def supervised_contrastive_loss(
    normalized: torch.Tensor, labels: torch.Tensor, temperature: float,
    include_same_prompt_negatives: bool,
) -> torch.Tensor:
    n = len(labels)
    if n != 2 * len(LABELS):
        raise ValueError("Contrastive batches must contain two full prompts")
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not torch.all(positive.sum(dim=1) == 1):
        raise RuntimeError("Every anchor must have exactly one cross-prompt positive")
    if include_same_prompt_negatives:
        denominator_mask = ~eye
    else:
        prompt_group = torch.arange(n, device=labels.device) // len(LABELS)
        denominator_mask = prompt_group[:, None].ne(prompt_group[None, :])
    if torch.any(positive & ~denominator_mask):
        raise RuntimeError("Contrastive denominator excludes a positive example")
    similarity = normalized @ normalized.T / temperature
    similarity = similarity.masked_fill(~denominator_mask, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    return -log_probability[positive].mean()


def pad_rows(
    tokenizer, tokenized: dict[str, list[list[int]]], row_indices: np.ndarray,
    labels: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded = tokenizer.pad(
        [
            {"input_ids": tokenized["input_ids"][int(i)],
             "attention_mask": tokenized["attention_mask"][int(i)]}
            for i in row_indices
        ],
        padding=True, return_tensors="pt",
    )
    return (
        padded["input_ids"].to(device),
        padded["attention_mask"].to(device),
        torch.as_tensor(labels[row_indices], dtype=torch.long, device=device),
    )


def evaluate(
    model: PromptClassifier, tokenizer, tokenized: dict[str, list[list[int]]],
    labels: np.ndarray, indices: np.ndarray, batch_size: int,
    device: torch.device, scores: np.ndarray,
) -> None:
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            batch_rows = indices[offset: offset + batch_size]
            input_ids, attention_mask, _ = pad_rows(
                tokenizer, tokenized, batch_rows, labels, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(input_ids, attention_mask)
            scores[batch_rows] = logits.float().cpu().numpy()


def run_condition(
    name: str, contrastive_weight: float, include_same_prompt_negatives: bool,
    args: argparse.Namespace,
    dev: pd.DataFrame, tokenizer, tokenized: dict[str, list[list[int]]],
    labels: np.ndarray, prompt_rows: dict[str, np.ndarray],
    schedules: dict[int, list[list[tuple[str, str]]]], device: torch.device,
) -> dict[str, object]:
    output = args.output_dir / name
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Condition output already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []

    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        fit_idx = np.flatnonzero(dev["fold"].to_numpy() != str(fold))
        valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(fold))
        seed_everything(args.seed)
        model = PromptClassifier(args.dropout).to(device)
        revision = getattr(model.encoder.config, "_commit_hash", None)
        if revision != REVISION:
            raise RuntimeError(f"Loaded MiniLM revision {revision!r} differs from {REVISION}")
        class_counts = np.bincount(labels[fit_idx], minlength=len(LABELS))
        if np.any(class_counts == 0) or set(labels[valid_idx]) != set(range(len(LABELS))):
            raise ValueError(f"Fold {fold} lacks one or more labels")
        weights = len(fit_idx) / (len(LABELS) * class_counts.astype(np.float32))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate},
                {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        total_steps = args.epochs * len(schedules[fold][0])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=round(args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda")
        epoch_rows: list[dict[str, float | int | None]] = []
        skipped_steps = 0
        for epoch_number, prompt_pairs in enumerate(schedules[fold], start=1):
            model.train()
            total_loss_sum = 0.0
            ce_loss_sum = 0.0
            con_loss_sum = 0.0
            examples_seen = 0
            for first_prompt, second_prompt in prompt_pairs:
                batch_rows = np.concatenate((prompt_rows[first_prompt], prompt_rows[second_prompt]))
                if len(batch_rows) != 36:
                    raise RuntimeError("Expected exactly 36 rows from two full prompts")
                input_ids, attention_mask, batch_labels = pad_rows(
                    tokenizer, tokenized, batch_rows, labels, device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, normalized = model(input_ids, attention_mask)
                    ce_loss = F.cross_entropy(logits.float(), batch_labels, weight=class_weights)
                    if contrastive_weight > 0:
                        contrastive_loss = supervised_contrastive_loss(
                            normalized, batch_labels, args.temperature,
                            include_same_prompt_negatives,
                        )
                        loss = ce_loss + contrastive_weight * contrastive_loss
                    else:
                        contrastive_loss = None
                        loss = ce_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                else:
                    skipped_steps += 1
                count = len(batch_rows)
                total_loss_sum += float(loss.item()) * count
                ce_loss_sum += float(ce_loss.item()) * count
                if contrastive_loss is not None:
                    con_loss_sum += float(contrastive_loss.item()) * count
                examples_seen += count
            epoch_row: dict[str, float | int | None] = {
                "epoch": epoch_number,
                "total_loss": total_loss_sum / examples_seen,
                "ce_loss": ce_loss_sum / examples_seen,
                "contrastive_loss": (con_loss_sum / examples_seen) if contrastive_weight else None,
                "skipped_optimizer_steps_cumulative": skipped_steps,
            }
            epoch_rows.append(epoch_row)
            print(
                f"{name} fold={fold} epoch={epoch_number}/{args.epochs} "
                f"total_loss={epoch_row['total_loss']:.6f} "
                f"ce_loss={epoch_row['ce_loss']:.6f}", flush=True,
            )

        evaluate(
            model, tokenizer, tokenized, labels, valid_idx,
            args.validation_batch_size, device, scores,
        )
        predicted = scores[valid_idx].argmax(axis=1)
        fold_score = float(balanced_accuracy_score(labels[valid_idx], predicted))
        fold_row: dict[str, object] = {
            "fold": fold,
            "train_rows": len(fit_idx),
            "validation_rows": len(valid_idx),
            "train_prompts": int(dev.iloc[fit_idx]["prompt_id"].nunique()),
            "validation_prompts": int(dev.iloc[valid_idx]["prompt_id"].nunique()),
            "balanced_accuracy": fold_score,
            "epoch_losses": epoch_rows,
            "skipped_optimizer_steps": skipped_steps,
            "batch_order_sha256": json_sha256(schedules[fold]),
            "duration_seconds": time.perf_counter() - fold_started,
        }
        fold_rows.append(fold_row)
        torch.save(model.cpu().state_dict(), output / f"fold_{fold}_state_dict.pt")
        print(f"{name} fold={fold} BA={fold_score:.6f}", flush=True)
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    if np.isnan(scores).any():
        raise RuntimeError(f"Incomplete OOF scores for {name}")
    predicted = scores.argmax(axis=1)
    oof_score = float(balanced_accuracy_score(labels, predicted))
    confusion = confusion_matrix(labels, predicted, labels=np.arange(len(LABELS)))
    class_metrics = pd.DataFrame({
        "model": LABELS,
        "support": confusion.sum(axis=1),
        "correct": confusion.diagonal(),
    })
    class_metrics["recall"] = class_metrics["correct"] / class_metrics["support"]
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = dev["model"]
    oof["pred_model"] = [LABELS[index] for index in predicted]
    oof["fold"] = dev["fold"]
    oof["correct"] = labels == predicted
    for index, label in enumerate(LABELS):
        oof[f"score_{label}"] = scores[:, index]
    oof.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "epoch_losses"}
        for row in fold_rows
    ]).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame([
        {"fold": row["fold"], **epoch_row}
        for row in fold_rows for epoch_row in row["epoch_losses"]
    ]).to_csv(output / "epoch_losses.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(output / "confusion_matrix.csv")
    class_metrics.to_csv(output / "class_metrics.csv", index=False)

    metrics: dict[str, object] = {
        "experiment": "e4_prompt_contrastive",
        "condition": name,
        "model_id": MODEL_ID,
        "pretrained_revision": REVISION,
        "architecture": "MiniLM masked mean pooling; dropout+linear CE head; L2-normalized pooled representation for SupCon",
        "batch_design": "two complete prompts per batch; 36 responses; one cross-prompt positive per anchor",
        "contrastive_negative_set": (
            "17 same-prompt plus 17 cross-prompt negatives"
            if include_same_prompt_negatives else "17 cross-prompt negatives only"
        ) if contrastive_weight else "not applicable",
        "contrastive_weight": contrastive_weight,
        "temperature": args.temperature,
        "oof_balanced_accuracy": oof_score,
        "mean_fold_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "fold_metrics": fold_rows,
        "n_rows_development": len(dev),
        "n_prompts_development": int(dev["prompt_id"].nunique()),
        "n_prompts_holdout_untouched": 30,
        "labels": list(LABELS),
        "epochs": args.epochs,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "max_length": args.max_length,
        "batch_size": 36,
        "validation_batch_size": args.validation_batch_size,
        "seed": args.seed,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "class_weights": "inverse fold-training class frequency",
        "epoch_selection": "fixed final epoch; no outer-validation selection",
        "optimizer": "AdamW",
        "scheduler": "linear decay with warmup; advance only after actual optimizer step",
        "mixed_precision": "fp16 autocast with GradScaler",
        "train_file": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "e0_oof_sha256": sha256(args.e0_oof),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": driver_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "duration_seconds": time.perf_counter() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name} OOF BA={oof_score:.6f}", flush=True)
    return metrics


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.epochs != 12 or args.max_length != 256 or args.seed != 42:
        raise ValueError("This registered E4 comparison requires 12 epochs, 256 tokens, seed 42")
    if args.temperature != 0.1 or args.contrastive_weight != 0.2:
        raise ValueError("This registered E4 comparison requires temperature 0.1 and lambda 0.2")
    if args.encoder_learning_rate != 2e-5 or args.head_learning_rate != 1e-3:
        raise ValueError("This registered E4 comparison requires encoder/head LR 2e-5/1e-3")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")

    seed_everything(args.seed)
    dev = load_development(args.train, args.split_manifest, args.e0_oof)
    labels = np.asarray([LABELS.index(label) for label in dev["model"]], dtype=np.int64)
    prompt_rows = {
        prompt_id: np.asarray(sorted(group.index, key=lambda i: int(labels[i])), dtype=np.int64)
        for prompt_id, group in dev.groupby("prompt_id")
    }
    schedules = make_pair_schedule(dev, args.seed, args.epochs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
    tokens = tokenizer(dev["text"].tolist(), truncation=True, max_length=args.max_length, padding=False)
    tokenized = {key: tokens[key] for key in ("input_ids", "attention_mask")}
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "conditions": {
            name: {
                "contrastive_weight": weight,
                "include_same_prompt_negatives": include_same_prompt_negatives,
            }
            for name, weight, include_same_prompt_negatives in CONDITIONS
        },
        "selected_condition": args.only_condition or "all",
        "epochs": args.epochs,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "max_length": args.max_length,
        "batch_size": 36,
        "seed": args.seed,
        "temperature": args.temperature,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "holdout_prompts_untouched": 30,
        "train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "e0_oof_sha256": sha256(args.e0_oof),
        "epoch_selection": "fixed final epoch",
        "batch_order_sha256_by_fold": {
            str(fold): json_sha256(schedules[fold]) for fold in range(1, N_FOLDS + 1)
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([
        {
            "fold": fold,
            "epoch": epoch,
            "batch": batch,
            "first_prompt": first,
            "second_prompt": second,
        }
        for fold, epochs in schedules.items()
        for epoch, pairs in enumerate(epochs, start=1)
        for batch, (first, second) in enumerate(pairs, start=1)
    ]).to_csv(args.output_dir / "batch_pairs.csv", index=False)

    results = {}
    for name, weight, include_same_prompt_negatives in CONDITIONS:
        if args.only_condition and name != args.only_condition:
            continue
        results[name] = run_condition(
            name, weight, include_same_prompt_negatives,
            args, dev, tokenizer, tokenized, labels,
            prompt_rows, schedules, device,
        )["oof_balanced_accuracy"]
    print(f"E4 comparison: {results}", flush=True)


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
