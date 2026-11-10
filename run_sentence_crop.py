#!/usr/bin/env python3
"""One fixed sentence-window crop experiment on the original RoBERTa folds.

Training inputs alone can change. Validation and OOF fusion use original text.
The locked 30-prompt holdout is never tokenized, fitted, or scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_e0 import LABELS, sha256
from run_finetune_encoder import (
    MeanPoolClassifier, TokenizedDataset, driver_version,
    load_development, make_collate, seed_everything,
)
from sentence_crop import (
    CROP_PROBABILITY, MAX_RATIO, MIN_CROP_WORDS, MIN_ORIGINAL_WORDS,
    MIN_RATIO, candidate_spans, crop_for_epoch,
)


MODEL_ID = "FacebookAI/roberta-base"
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
SEED = 42
N_FOLDS = 5
EPOCHS = 12
MAX_LENGTH = 256
BATCH_SIZE = 16
ENCODER_LR = 2e-5
HEAD_LR = 1e-3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
FAMILY_NAMES = ("gpt", "gemini", "claude", "qwen", "glm", "kimi")
FAMILY_IDS = np.asarray([
    FAMILY_NAMES.index("qwen" if label.startswith("qwen") else label.split("-", 1)[0])
    for label in LABELS
], dtype=np.int64)


def prepare_epoch(
    dev: pd.DataFrame, fit_idx: np.ndarray, spans: list[list[tuple[int, int]]],
    original_encodings: dict[str, list[list[int]]], original_token_lengths: np.ndarray,
    tokenizer, fold: int, epoch: int,
) -> tuple[dict[str, list[list[int]]], list[dict[str, object]], dict[str, object]]:
    """Replace only cropped training rows; keep all other encodings untouched."""
    epoch_encodings = {key: values.copy() for key, values in original_encodings.items()}
    selected_rows: list[int] = []
    cropped_texts: list[str] = []
    crop_log: list[dict[str, object]] = []
    for index in fit_idx:
        row = dev.iloc[int(index)]
        source = row["text"]
        crop, did_crop = crop_for_epoch(
            source, spans[int(index)], seed=SEED, outer_fold=fold,
            epoch=epoch, row_id=row["id"],
        )
        if did_crop:
            if crop not in source or len(crop.split()) < MIN_CROP_WORDS:
                raise ValueError("Crop is not an original substring of sufficient length")
            selected_rows.append(int(index))
            cropped_texts.append(crop)
            crop_log.append({
                "fold": fold, "epoch": epoch, "id": row["id"],
                "original_word_count": len(source.split()),
                "cropped_word_count": len(crop.split()),
                "original_token_count": int(original_token_lengths[int(index)]),
            })
    if selected_rows:
        tokenized = tokenizer(
            cropped_texts, truncation=True, max_length=MAX_LENGTH, padding=False
        )
        full = tokenizer(cropped_texts, truncation=False, padding=False)
        for position, row_index in enumerate(selected_rows):
            for key in ("input_ids", "attention_mask"):
                epoch_encodings[key][row_index] = tokenized[key][position]
            crop_log[position]["cropped_token_count"] = len(full["input_ids"][position])
            if len(epoch_encodings["input_ids"][row_index]) > MAX_LENGTH:
                raise ValueError("A cropped training row exceeded the tokenizer limit")
    eligible = int(sum(bool(spans[int(index)]) for index in fit_idx))
    summary = {
        "fold": fold, "epoch": epoch, "train_rows": len(fit_idx),
        "eligible_rows": eligible, "cropped_rows": len(selected_rows),
        "below_40_words": int(sum(row["cropped_word_count"] < MIN_CROP_WORDS for row in crop_log)),
        "cropped_over_256_tokens": int(sum(row["cropped_token_count"] > MAX_LENGTH for row in crop_log)),
        "original_words_mean_cropped_rows": float(np.mean([
            row["original_word_count"] for row in crop_log])) if crop_log else None,
        "cropped_words_mean": float(np.mean([
            row["cropped_word_count"] for row in crop_log])) if crop_log else None,
        "original_tokens_mean_cropped_rows": float(np.mean([
            row["original_token_count"] for row in crop_log])) if crop_log else None,
        "cropped_tokens_mean": float(np.mean([
            row["cropped_token_count"] for row in crop_log])) if crop_log else None,
    }
    return epoch_encodings, crop_log, summary


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    started = time.perf_counter()
    seed_everything(SEED)
    dev = load_development(args.train, args.split_manifest, args.e0_oof)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Expected only the fixed 120-prompt development partition")
    labels = np.asarray([LABELS.index(label) for label in dev["model"]], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
    originals = tokenizer(dev["text"].tolist(), truncation=True,
                          max_length=MAX_LENGTH, padding=False)
    original_encodings = {key: originals[key] for key in ("input_ids", "attention_mask")}
    untruncated = tokenizer(dev["text"].tolist(), truncation=False, padding=False)
    original_token_lengths = np.asarray([len(ids) for ids in untruncated["input_ids"]], dtype=int)
    spans = [candidate_spans(text) for text in dev["text"]]
    if (sum(len(text.split()) >= MIN_ORIGINAL_WORDS for text in dev["text"]) != 871
            or sum(bool(candidates) for candidates in spans) != 863):
        raise ValueError("Pre-registered crop coverage 863/871 did not reproduce")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "round3_sentence_window_crop_single_factor",
        "model_id": MODEL_ID, "requested_revision": REVISION,
        "epochs": EPOCHS, "encoder_learning_rate": ENCODER_LR,
        "head_learning_rate": HEAD_LR, "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE, "seed": SEED, "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "pooling": "mean",
        "epoch_selection": "fixed final epoch",
        "train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "e0_oof_sha256": sha256(args.e0_oof),
        "crop_script_sha256": sha256(Path(__file__)),
        "crop_operator_sha256": sha256(Path("sentence_crop.py")),
        "base_trainer_script_sha256": sha256(Path("run_finetune_encoder.py")),
        "n_rows_development": len(dev),
        "n_prompts_development": int(dev["prompt_id"].nunique()),
        "holdout_scored": False,
        "crop_probability": CROP_PROBABILITY,
        "min_original_words": MIN_ORIGINAL_WORDS,
        "min_crop_words": MIN_CROP_WORDS,
        "min_ratio": MIN_RATIO, "max_ratio": MAX_RATIO,
        "eligible_original_rows": 863,
        "crop_rng": "SHA256(seed:outer_fold:epoch:id:crop), first 16 hex digits to random.Random",
        "crop_only_training_input": True,
        "validation_original_text": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame({
        "id": dev["id"], "word_count": [len(text.split()) for text in dev["text"]],
        "token_count": original_token_lengths,
    }).to_csv(args.output_dir / "development_lengths.csv", index=False)

    device = torch.device("cuda:0")
    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []
    crop_summaries: list[dict[str, object]] = []
    pretrained_revision = None
    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        fit_idx = np.flatnonzero(dev["fold"].to_numpy() != str(fold))
        valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(fold))
        if len(fit_idx) != 1728 or len(valid_idx) != 432:
            raise ValueError("Unexpected fixed fold sizes")
        seed_everything(SEED)
        model = MeanPoolClassifier(MODEL_ID, REVISION, DROPOUT, "mean").to(device)
        revision = getattr(model.encoder.config, "_commit_hash", None)
        if revision != REVISION or (pretrained_revision is not None and pretrained_revision != revision):
            raise RuntimeError("The pretrained RoBERTa revision changed")
        pretrained_revision = revision
        counts = np.bincount(labels[fit_idx], minlength=len(LABELS))
        class_weights = torch.tensor(
            len(fit_idx) / (len(LABELS) * counts.astype(np.float32)),
            dtype=torch.float32, device=device,
        )
        train_ds = TokenizedDataset(original_encodings, labels, fit_idx)
        valid_ds = TokenizedDataset(original_encodings, labels, valid_idx)
        fit_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
            collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=True,
        )
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.classifier.parameters(), "lr": HEAD_LR},
        ], weight_decay=WEIGHT_DECAY)
        total_steps = EPOCHS * len(fit_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=round(WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        epoch_losses: list[float] = []
        skipped = 0
        for epoch in range(1, EPOCHS + 1):
            epoch_encodings, epoch_crops, summary = prepare_epoch(
                dev, fit_idx, spans, original_encodings, original_token_lengths,
                tokenizer, fold, epoch,
            )
            train_ds.encodings = epoch_encodings
            crop_rows.extend(epoch_crops)
            crop_summaries.append(summary)
            model.train()
            loss_sum = 0.0
            n_seen = 0
            for batch in fit_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids, attention_mask).float()
                    loss = F.cross_entropy(logits, y, weight=class_weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                else:
                    skipped += 1
                loss_sum += float(loss.item()) * len(y)
                n_seen += len(y)
            epoch_losses.append(loss_sum / n_seen)
            print(f"fold={fold} epoch={epoch}/{EPOCHS} "
                  f"crop={summary['cropped_rows']} loss={epoch_losses[-1]:.6f}", flush=True)
        model.eval()
        with torch.inference_mode():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids, attention_mask)
                scores[batch["row_index"].numpy()] = logits.float().cpu().numpy()
        predicted = scores[valid_idx].argmax(axis=1)
        family_correct = FAMILY_IDS[predicted] == FAMILY_IDS[labels[valid_idx]]
        fold_rows.append({
            "fold": fold, "train_rows": len(fit_idx), "validation_rows": len(valid_idx),
            "train_prompts": int(dev.iloc[fit_idx]["prompt_id"].nunique()),
            "validation_prompts": int(dev.iloc[valid_idx]["prompt_id"].nunique()),
            "balanced_accuracy": float(balanced_accuracy_score(labels[valid_idx], predicted)),
            "family_accuracy": float(family_correct.mean()),
            "within_family_accuracy": float((predicted == labels[valid_idx]).sum() / family_correct.sum()),
            "epoch_train_loss": epoch_losses,
            "skipped_optimizer_steps": skipped,
            "cropped_training_events": sum(row["cropped_rows"] for row in crop_summaries if row["fold"] == fold),
            "duration_seconds": time.perf_counter() - fold_started,
        })
        print(f"fold={fold} balanced_accuracy={fold_rows[-1]['balanced_accuracy']:.6f}", flush=True)
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    if np.isnan(scores).any():
        raise RuntimeError("Crop OOF scores are incomplete")
    if any(row["below_40_words"] for row in crop_summaries):
        raise RuntimeError("A crop fell below the minimum word count")
    predicted = scores.argmax(axis=1)
    confusion = confusion_matrix(labels, predicted, labels=np.arange(len(LABELS)))
    family_correct = FAMILY_IDS[predicted] == FAMILY_IDS[labels]
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = dev["model"]
    oof["pred_model"] = [LABELS[index] for index in predicted]
    oof["fold"] = dev["fold"]
    oof["correct"] = labels == predicted
    for index, label in enumerate(LABELS):
        oof[f"score_{label}"] = scores[:, index]
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(crop_summaries).to_csv(args.output_dir / "crop_epoch_metrics.csv", index=False)
    pd.DataFrame(crop_rows).to_csv(args.output_dir / "crop_events.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model").to_csv(args.output_dir / "confusion_matrix.csv")
    pd.DataFrame({
        "model": LABELS, "support": confusion.sum(axis=1),
        "correct": confusion.diagonal(),
        "recall": confusion.diagonal() / confusion.sum(axis=1),
    }).to_csv(args.output_dir / "class_metrics.csv", index=False)
    metrics = {
        **config,
        "pretrained_revision": pretrained_revision,
        "oof_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "oof_family_accuracy": float(family_correct.mean()),
        "oof_within_family_accuracy": float((predicted == labels).sum() / family_correct.sum()),
        "fold_metrics": fold_rows,
        "cropped_training_events": len(crop_rows),
        "gpu_name": torch.cuda.get_device_name(device),
        "nvidia_driver_version": driver_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "duration_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"OOF balanced_accuracy={metrics['oof_balanced_accuracy']:.6f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("outputs/e0_tfidf_svm/split_manifest.csv"))
    parser.add_argument("--e0-oof", type=Path,
                        default=Path("outputs/e0_tfidf_svm/oof_predictions.csv"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
