#!/usr/bin/env python3
"""E2: fixed-schedule MiniLM supervised fine-tuning on E0's prompt-grouped folds.

The locked prompt holdout is never tokenized, trained on, or evaluated here.
Run with CUDA_VISIBLE_DEVICES=1 to target physical GPU 1 on the research server.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from run_e0 import LABELS, load_and_validate


DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
N_FOLDS = 5
DEFAULT_SEED = 42
FAMILY_NAMES = ("gpt", "gemini", "claude", "qwen", "glm", "kimi")


def family(label: str) -> str:
    if label.startswith("qwen"):
        return "qwen"
    return label.split("-", 1)[0]


LABEL_FAMILY_IDS = tuple(FAMILY_NAMES.index(family(label)) for label in LABELS)
FAMILY_LEAF_INDICES = tuple(
    tuple(index for index, family_id in enumerate(LABEL_FAMILY_IDS) if family_id == group)
    for group in range(len(FAMILY_NAMES))
)


def aggregate_family_logits(leaf_logits: torch.Tensor) -> torch.Tensor:
    """Log of each family's summed leaf softmax mass, up to one common constant."""
    return torch.stack(
        [torch.logsumexp(leaf_logits[:, indices], dim=1) for indices in FAMILY_LEAF_INDICES],
        dim=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision", default=None,
        help="Exact Hugging Face commit SHA to pin tokenizer and encoder",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("outputs/e0_tfidf_svm/split_manifest.csv"),
    )
    parser.add_argument(
        "--e0-oof", type=Path,
        default=Path("outputs/e0_tfidf_svm/oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/e2_finetune/minilm"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling", choices=("normalized_mean", "mean"), default="normalized_mean",
        help="Pool token embeddings by their masked mean, optionally L2-normalized",
    )
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument(
        "--family-loss-weight", type=float, default=0.0,
        help="Nonnegative family CE weight; the default uses 18-class logits (default: 0)",
    )
    parser.add_argument(
        "--independent-family-head", action="store_true",
        help="Use a separate 6-family head for family CE; 18-class head remains the inference output",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development(
    train_path: Path, manifest_path: Path, e0_oof_path: Path
) -> pd.DataFrame:
    frame = load_and_validate(train_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("Expected prompt_id,split columns in E0 split manifest")
    if manifest["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompts in split manifest")
    if set(manifest["prompt_id"]) != set(frame["prompt_id"]):
        raise ValueError("Split manifest prompts differ from train.csv")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("Split manifest has unexpected split names")
    if int((manifest["split"] == "holdout").sum()) != 30:
        raise ValueError("Expected exactly 30 locked holdout prompts")

    oof = pd.read_csv(e0_oof_path, dtype=str, keep_default_na=False)
    required = {"id", "prompt_id", "true_model", "fold"}
    if not required.issubset(oof.columns):
        raise ValueError(f"E0 OOF is missing columns {sorted(required - set(oof.columns))}")
    if oof["id"].duplicated().any():
        raise ValueError("Duplicate ids in E0 OOF")
    dev_prompts = set(manifest.loc[manifest["split"] == "development", "prompt_id"])
    dev = frame.loc[frame["prompt_id"].isin(dev_prompts)].copy()
    if set(oof["id"]) != set(dev["id"]):
        raise ValueError("E0 OOF ids differ from development ids")

    oof = oof.loc[:, ["id", "prompt_id", "true_model", "fold"]]
    dev = dev.merge(oof, on="id", validate="one_to_one", suffixes=("", "_e0"))
    if not dev["prompt_id"].equals(dev["prompt_id_e0"]):
        raise ValueError("E0 OOF prompt IDs differ from train.csv")
    if not dev["model"].equals(dev["true_model"]):
        raise ValueError("E0 OOF labels differ from train.csv")
    if set(dev["fold"]) != {str(i) for i in range(1, N_FOLDS + 1)}:
        raise ValueError("Expected E0 folds 1..5")
    if dev.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("A prompt occurs in multiple E0 folds")
    for fold in range(1, N_FOLDS + 1):
        train_labels = set(dev.loc[dev["fold"] != str(fold), "model"])
        valid_labels = set(dev.loc[dev["fold"] == str(fold), "model"])
        if train_labels != set(LABELS) or valid_labels != set(LABELS):
            raise ValueError(f"Fold {fold} lacks one or more classes")
    return dev.reset_index(drop=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.splitlines()[0].strip() or None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


class TokenizedDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: np.ndarray, indices: np.ndarray):
        self.encodings = encodings
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = int(self.indices[item])
        result: dict[str, object] = {key: values[row] for key, values in self.encodings.items()}
        result["labels"] = int(self.labels[row])
        result["row_index"] = row
        return result


class MeanPoolClassifier(nn.Module):
    def __init__(
        self, model_id: str, revision: str | None, dropout: float, pooling: str,
        independent_family_head: bool = False,
    ):
        super().__init__()
        model_kwargs: dict[str, object] = {"revision": revision}
        # RoBERTa's optional pooler is absent from this checkpoint and unused by
        # the masked-mean classifier. Avoid initializing an untrained component.
        if model_id == "FacebookAI/roberta-base":
            model_kwargs["add_pooling_layer"] = False
        self.encoder = AutoModel.from_pretrained(model_id, **model_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))
        # Initialize after the leaf head so its seed and weights match the
        # original single-head run when the new option is enabled.
        self.family_classifier = (
            nn.Linear(self.encoder.config.hidden_size, len(FAMILY_NAMES))
            if independent_family_head else None
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        return_family_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if self.pooling == "normalized_mean":
            pooled = F.normalize(pooled, p=2, dim=1)
        representation = self.dropout(pooled)
        leaf_logits = self.classifier(representation)
        if return_family_logits:
            if self.family_classifier is None:
                raise RuntimeError("An independent family head was not initialized")
            return leaf_logits, self.family_classifier(representation)
        return leaf_logits


def make_collate(tokenizer):
    def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        padded = tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in batch],
            padding=True,
            return_tensors="pt",
        )
        padded["labels"] = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
        padded["row_index"] = torch.tensor([row["row_index"] for row in batch], dtype=torch.long)
        return padded
    return collate


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA build compatible with this server's driver")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("epochs, batch-size, and max-length must be positive")
    if args.learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("learning-rate and head-learning-rate must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if not np.isfinite(args.family_loss_weight) or args.family_loss_weight < 0:
        raise ValueError("family-loss-weight must be finite and nonnegative")
    if args.independent_family_head and args.family_loss_weight == 0:
        raise ValueError("independent-family-head requires a positive family-loss-weight")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")

    started = time.perf_counter()
    seed_everything(args.seed)
    dev = load_development(args.train, args.split_manifest, args.e0_oof)
    labels = np.asarray([LABELS.index(label) for label in dev["model"]], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, use_fast=True
    )
    tokenized = tokenizer(dev["text"].tolist(), truncation=True, max_length=args.max_length, padding=False)
    encodings = {key: tokenized[key] for key in ("input_ids", "attention_mask")}
    device = torch.device("cuda:0")
    family_ids = torch.tensor(LABEL_FAMILY_IDS, dtype=torch.long, device=device)
    family_loss_mode = (
        "independent_6_class_head" if args.independent_family_head
        else "leaf_logsumexp" if args.family_loss_weight else "none"
    )
    use_amp = True
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps({
            "model_id": args.model_id,
            "requested_revision": args.revision,
            "epochs": args.epochs,
            "encoder_learning_rate": args.learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "pooling": args.pooling,
            "family_loss_weight": args.family_loss_weight,
            "independent_family_head": args.independent_family_head,
            "family_loss_mode": family_loss_mode,
            "family_names": FAMILY_NAMES,
            "family_leaf_indices": FAMILY_LEAF_INDICES,
            "family_loss": (
                "none" if not args.family_loss_weight else
                "CE on a separate 6-family head; no hard routing" if args.independent_family_head else
                "CE on logsumexp of the 18 leaf logits within each family; no hard routing"
            ),
            "architecture": (
                "AutoModel last_hidden_state, attention-mask mean pooling, optional L2 normalization, "
                "dropout, linear 18-class head"
                + (", independent linear 6-family head" if args.independent_family_head else "")
            ),
            "roberta_pooler_disabled": args.model_id == "FacebookAI/roberta-base",
            "train_sha256": sha256(args.train),
            "split_manifest_sha256": sha256(args.split_manifest),
            "e0_oof_sha256": sha256(args.e0_oof),
            "n_rows_development": len(dev),
            "n_prompts_development": int(dev["prompt_id"].nunique()),
            "holdout_prompts_untouched": 30,
            "epoch_selection": "fixed final epoch",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    family_scores = (
        np.full((len(dev), len(FAMILY_NAMES)), np.nan, dtype=np.float32)
        if args.independent_family_head else None
    )
    fold_rows: list[dict[str, object]] = []
    pretrained_revision = None
    encoder_model_type = None
    encoder_hidden_size = None
    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        fit_idx = np.flatnonzero(dev["fold"].to_numpy() != str(fold))
        valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(fold))
        # Recreate the encoder and randomly initialized classifier for each outer fold.
        seed_everything(args.seed)
        model = MeanPoolClassifier(
            args.model_id, args.revision, args.dropout, args.pooling,
            independent_family_head=args.independent_family_head,
        ).to(device)
        revision = getattr(model.encoder.config, "_commit_hash", None)
        if encoder_model_type is None:
            encoder_model_type = model.encoder.config.model_type
            encoder_hidden_size = model.encoder.config.hidden_size
        if args.revision is not None and revision != args.revision:
            raise RuntimeError(
                f"Loaded revision {revision!r} does not match pinned SHA {args.revision!r}"
            )
        if pretrained_revision is None:
            pretrained_revision = revision
        elif pretrained_revision != revision:
            raise RuntimeError("The pretrained checkpoint revision changed between folds")

        class_counts = np.bincount(labels[fit_idx], minlength=len(LABELS))
        weights = len(fit_idx) / (len(LABELS) * class_counts.astype(np.float32))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        train_ds = TokenizedDataset(encodings, labels, fit_idx)
        valid_ds = TokenizedDataset(encodings, labels, valid_idx)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, generator=generator,
            collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=True,
        )
        parameter_groups = [
            {"params": model.encoder.parameters(), "lr": args.learning_rate},
            {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
        ]
        if model.family_classifier is not None:
            parameter_groups.append({
                "params": model.family_classifier.parameters(),
                "lr": args.head_learning_rate,
            })
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
        total_steps = args.epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=round(args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        epoch_losses: list[float] = []
        skipped_optimizer_steps = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            n_seen = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    if args.independent_family_head:
                        logits, independent_family_logits = model(
                            input_ids, attention_mask, return_family_logits=True
                        )
                    else:
                        logits = model(input_ids, attention_mask)
                    leaf_logits = logits.float()
                    loss = F.cross_entropy(leaf_logits, y, weight=class_weights)
                    if args.family_loss_weight:
                        coarse_logits = (
                            independent_family_logits.float() if args.independent_family_head
                            else aggregate_family_logits(leaf_logits)
                        )
                        coarse_loss = F.cross_entropy(coarse_logits, family_ids[y])
                        loss = loss + args.family_loss_weight * coarse_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                else:
                    skipped_optimizer_steps += 1
                loss_sum += float(loss.item()) * len(y)
                n_seen += len(y)
            epoch_losses.append(loss_sum / n_seen)
            print(f"fold={fold} epoch={epoch}/{args.epochs} train_loss={epoch_losses[-1]:.6f}", flush=True)

        # The fixed final epoch is evaluated once. Outer validation never chooses an epoch.
        model.eval()
        with torch.inference_mode():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    if args.independent_family_head:
                        logits, independent_family_logits = model(
                            input_ids, attention_mask, return_family_logits=True
                        )
                    else:
                        logits = model(input_ids, attention_mask)
                scores[batch["row_index"].numpy()] = logits.float().cpu().numpy()
                if family_scores is not None:
                    family_scores[batch["row_index"].numpy()] = (
                        independent_family_logits.float().cpu().numpy()
                    )
        predicted = scores[valid_idx].argmax(axis=1)
        fold_score = float(balanced_accuracy_score(labels[valid_idx], predicted))
        fold_family_correct = (
            np.asarray(LABEL_FAMILY_IDS)[predicted]
            == np.asarray(LABEL_FAMILY_IDS)[labels[valid_idx]]
        )
        fold_rows.append({
            "fold": fold,
            "train_rows": len(fit_idx),
            "validation_rows": len(valid_idx),
            "train_prompts": int(dev.iloc[fit_idx]["prompt_id"].nunique()),
            "validation_prompts": int(dev.iloc[valid_idx]["prompt_id"].nunique()),
            "balanced_accuracy": fold_score,
            "family_accuracy": float(fold_family_correct.mean()),
            "within_family_accuracy": float(
                (predicted == labels[valid_idx]).sum() / fold_family_correct.sum()
            ),
            "epoch_train_loss": epoch_losses,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "duration_seconds": time.perf_counter() - fold_started,
        })
        if family_scores is not None:
            fold_rows[-1]["independent_family_head_accuracy"] = float(
                (family_scores[valid_idx].argmax(axis=1)
                 == np.asarray(LABEL_FAMILY_IDS)[labels[valid_idx]]).mean()
            )
        if not args.no_save_checkpoints:
            torch.save(model.cpu().state_dict(), output / f"fold_{fold}_state_dict.pt")
        print(f"fold={fold} balanced_accuracy={fold_score:.6f} seconds={fold_rows[-1]['duration_seconds']:.1f}", flush=True)
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    if np.isnan(scores).any():
        raise RuntimeError("OOF scores are incomplete")
    if family_scores is not None and np.isnan(family_scores).any():
        raise RuntimeError("Independent family-head OOF scores are incomplete")
    predicted = scores.argmax(axis=1)
    oof_score = float(balanced_accuracy_score(labels, predicted))
    family_correct = (
        np.asarray(LABEL_FAMILY_IDS)[predicted]
        == np.asarray(LABEL_FAMILY_IDS)[labels]
    )
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
    if family_scores is not None:
        family_oof = dev.loc[:, ["id", "prompt_id"]].copy()
        family_oof["true_model"] = dev["model"]
        family_oof["fold"] = dev["fold"]
        family_oof["true_family"] = [FAMILY_NAMES[index] for index in np.asarray(LABEL_FAMILY_IDS)[labels]]
        family_oof["pred_family"] = np.asarray(FAMILY_NAMES)[family_scores.argmax(axis=1)]
        for index, name in enumerate(FAMILY_NAMES):
            family_oof[f"score_family_{name}"] = family_scores[:, index]
        family_oof.to_csv(output / "family_head_oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(output / "confusion_matrix.csv")
    class_metrics.to_csv(output / "class_metrics.csv", index=False)
    metrics = {
        "experiment": "e2_finetune_encoder",
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "pretrained_revision": pretrained_revision,
        "encoder_model_type": encoder_model_type,
        "encoder_hidden_size": encoder_hidden_size,
        "roberta_pooler_disabled": args.model_id == "FacebookAI/roberta-base",
        "architecture": (
            f"AutoModel {args.pooling} pool, dropout, linear 18-class head"
            + (", independent linear 6-family head" if args.independent_family_head else "")
        ),
        "metric": "balanced_accuracy",
        "oof_balanced_accuracy": oof_score,
        "oof_family_accuracy": float(family_correct.mean()),
        "oof_within_family_accuracy": float((predicted == labels).sum() / family_correct.sum()),
        "mean_fold_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "fold_metrics": fold_rows,
        "n_rows_development": len(dev),
        "n_prompts_development": int(dev["prompt_id"].nunique()),
        "n_prompts_holdout_untouched": 30,
        "n_classes": len(LABELS),
        "labels": list(LABELS),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "family_loss_weight": args.family_loss_weight,
        "independent_family_head": args.independent_family_head,
        "family_loss_mode": family_loss_mode,
        "family_names": FAMILY_NAMES,
        "family_loss": (
            "none" if not args.family_loss_weight else
            "CE on independent 6-family head; 18-class head remains the inference output"
            if args.independent_family_head else
            "CE on summed leaf probability per family; 18-class head remains the inference output"
        ),
        "class_weights": "inverse fold-training class frequency",
        "epoch_selection": "fixed final epoch; no outer-validation selection",
        "optimizer": "AdamW",
        "scheduler": "linear decay with warmup; advances only after an optimizer step",
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
    if family_scores is not None:
        metrics["oof_independent_family_head_accuracy"] = float(
            (family_scores.argmax(axis=1) == np.asarray(LABEL_FAMILY_IDS)[labels]).mean()
        )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OOF balanced_accuracy={oof_score:.6f}", flush=True)
    print(f"Artifacts: {output.resolve()}", flush=True)


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
