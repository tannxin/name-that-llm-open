#!/usr/bin/env python3
"""One fixed within-family margin experiment with fold-internal E0c pair selection.

Stage 1 fits fresh E0c pipelines only on each outer training fold. Stage 2
trains RoBERTa only after verifying the saved inner OOF and its provenance.
The locked 30-prompt holdout and saved full-development E10/E11 predictions
are never used for pair selection or training.
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
import sklearn
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_combined import make_pipeline as make_e0c_pipeline
from run_e0 import LABELS, sha256
from run_finetune_encoder import (
    MeanPoolClassifier, TokenizedDataset, driver_version,
    load_development, make_collate, seed_everything,
)


MODEL_ID = "FacebookAI/roberta-base"
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
SEED = 42
N_OUTER = 5
N_INNER = 3
N_SELECTED_PAIRS = 2
EPOCHS = 12
MAX_LENGTH = 256
BATCH_SIZE = 16
ENCODER_LR = 2e-5
HEAD_LR = 1e-3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
MARGIN_WEIGHT = 0.1
MARGIN = 0.5
FAMILY_NAMES = ("gpt", "gemini", "claude", "qwen", "glm", "kimi")


def family(label: str) -> str:
    return "qwen" if label.startswith("qwen") else label.split("-", 1)[0]


FAMILY_IDS = np.asarray([FAMILY_NAMES.index(family(label)) for label in LABELS], dtype=np.int64)
PAIRS = tuple(
    (left, right) for left in range(len(LABELS))
    for right in range(left + 1, len(LABELS))
    if FAMILY_IDS[left] == FAMILY_IDS[right]
)
assert len(PAIRS) == 27


def paths_and_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "e0_oof_sha256": sha256(args.e0_oof),
        "margin_script_sha256": sha256(Path(__file__)),
        "e0c_pipeline_script_sha256": sha256(Path("run_combined.py")),
        "style_script_sha256": sha256(Path("run_style.py")),
        "base_trainer_script_sha256": sha256(Path("run_finetune_encoder.py")),
    }


def check_development(dev: pd.DataFrame) -> None:
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Expected the fixed 120-prompt, 2,160-row development data")
    if (dev.groupby("prompt_id")["model"].nunique() != len(LABELS)).any():
        raise ValueError("Each development prompt must contain all 18 labels")
    if not (dev.groupby("model").size() == 120).all():
        raise ValueError("Expected 120 development rows per label")


def select_pairs(confusion: np.ndarray) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    ranked = sorted(
        PAIRS,
        key=lambda pair: (
            -int(confusion[pair[0], pair[1]] + confusion[pair[1], pair[0]]),
            pair[0], pair[1],
        ),
    )
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    for left, right in ranked:
        if left not in used and right not in used:
            chosen.append((left, right))
            used.update((left, right))
        if len(chosen) == N_SELECTED_PAIRS:
            break
    if len(chosen) != N_SELECTED_PAIRS:
        raise RuntimeError("Could not choose two disjoint family pairs")
    rows = [{
        "left_model": LABELS[left],
        "right_model": LABELS[right],
        "left_to_right": int(confusion[left, right]),
        "right_to_left": int(confusion[right, left]),
        "mutual_errors": int(confusion[left, right] + confusion[right, left]),
        "selected": (left, right) in chosen,
    } for left, right in PAIRS]
    return chosen, rows


def prepare_selection(args: argparse.Namespace, dev: pd.DataFrame) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Selection output directory is not empty: {args.output_dir}")
    started = time.perf_counter()
    inner_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for outer in range(1, N_OUTER + 1):
        outer_train = dev.loc[dev["fold"].ne(str(outer))].reset_index(drop=True)
        outer_valid_prompts = set(dev.loc[dev["fold"].eq(str(outer)), "prompt_id"])
        if len(outer_train) != 1728 or outer_train["prompt_id"].nunique() != 96:
            raise ValueError(f"Outer fold {outer} has unexpected training size")
        if set(outer_train["prompt_id"]) & outer_valid_prompts:
            raise ValueError("Outer validation prompt entered inner selection")
        texts = outer_train["text"].to_numpy()
        labels = outer_train["model"].to_numpy()
        prompts = outer_train["prompt_id"].to_numpy()
        predicted = np.full(len(outer_train), None, dtype=object)
        inner_assignment = np.zeros(len(outer_train), dtype=np.int8)
        inner_split = GroupKFold(n_splits=N_INNER).split(texts, labels, prompts)
        for inner, (fit_idx, valid_idx) in enumerate(inner_split, start=1):
            fit_prompts = set(prompts[fit_idx])
            valid_prompts = set(prompts[valid_idx])
            if fit_prompts & valid_prompts or len(fit_prompts) != 64 or len(valid_prompts) != 32:
                raise ValueError("Inner folds are not prompt-disjoint 64/32 splits")
            if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
                raise ValueError("An inner fold lacks an official class")
            # A fresh E0c pipeline fits all TF-IDF vocabulary, style scaling,
            # normalization, and SVM parameters using only these 64 prompts.
            pipeline = make_e0c_pipeline()
            pipeline.fit(texts[fit_idx], labels[fit_idx])
            predicted[valid_idx] = pipeline.predict(texts[valid_idx])
            inner_assignment[valid_idx] = inner
            print(
                f"outer={outer} inner={inner}/{N_INNER} "
                f"fit_prompts={len(fit_prompts)} valid_prompts={len(valid_prompts)}",
                flush=True,
            )
        if any(value is None for value in predicted) or (inner_assignment == 0).any():
            raise RuntimeError("Inner OOF predictions are incomplete")
        confusion = confusion_matrix(labels, predicted, labels=LABELS)
        chosen, rows = select_pairs(confusion)
        for row in rows:
            pair_rows.append({"outer_fold": outer, **row})
        for index, row in outer_train.iterrows():
            inner_rows.append({
                "outer_fold": outer,
                "inner_fold": int(inner_assignment[index]),
                "id": row["id"],
                "prompt_id": row["prompt_id"],
                "true_model": row["model"],
                "pred_model": predicted[index],
            })
        fold_rows.append({
            "outer_fold": outer,
            "inner_oof_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "selected_pairs": [[LABELS[left], LABELS[right]] for left, right in chosen],
        })
        print(f"outer={outer} selected_pairs={fold_rows[-1]['selected_pairs']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inner_path = args.output_dir / "inner_oof_predictions.csv"
    pair_path = args.output_dir / "pair_counts.csv"
    pd.DataFrame(inner_rows).to_csv(inner_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    config = {
        "stage": "selection",
        "selector": "fresh E0c three-fold GroupKFold OOF inside each outer training fold",
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
        "n_pairs": N_SELECTED_PAIRS,
        "pair_rule": "top mutual confusion, LABELS-index tie break, greedy disjoint",
        "selected_by_outer_fold": fold_rows,
        "n_inner_oof_rows": len(inner_rows),
        "holdout_used": False,
        "full_development_e10_e11_oof_used": False,
        "inner_oof_sha256": sha256(inner_path),
        "pair_counts_sha256": sha256(pair_path),
        "duration_seconds": time.perf_counter() - started,
        "sklearn_version": sklearn.__version__,
        **paths_and_hashes(args),
    }
    (args.output_dir / "selection_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Selection artifacts: {args.output_dir.resolve()}", flush=True)


def verified_selection(args: argparse.Namespace, dev: pd.DataFrame) -> dict[int, list[tuple[int, int]]]:
    config_path = args.selection_dir / "selection_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in paths_and_hashes(args).items():
        if config.get(key) != value:
            raise ValueError(f"Selection provenance mismatch: {key}")
    inner_path = args.selection_dir / "inner_oof_predictions.csv"
    pair_path = args.selection_dir / "pair_counts.csv"
    if sha256(inner_path) != config["inner_oof_sha256"]:
        raise ValueError("Inner OOF file changed since selection")
    if sha256(pair_path) != config["pair_counts_sha256"]:
        raise ValueError("Pair-count file changed since selection")
    inner = pd.read_csv(inner_path, dtype={
        "outer_fold": int, "inner_fold": int, "id": str,
        "prompt_id": str, "true_model": str, "pred_model": str,
    })
    pair_table = pd.read_csv(pair_path, dtype={"outer_fold": int})
    if len(inner) != N_OUTER * 1728 or len(pair_table) != N_OUTER * len(PAIRS):
        raise ValueError("Selection artifacts have unexpected row counts")
    selections: dict[int, list[tuple[int, int]]] = {}
    for outer in range(1, N_OUTER + 1):
        fit = dev.loc[dev["fold"].ne(str(outer))].reset_index(drop=True)
        part = inner.loc[inner["outer_fold"].eq(outer)].reset_index(drop=True)
        expected = fit.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
        if not expected.equals(part.loc[:, ["id", "prompt_id", "true_model"]]):
            raise ValueError(f"Inner OOF rows do not exactly equal outer {outer} training rows")
        if part["id"].duplicated().any() or not set(part["pred_model"]).issubset(LABELS):
            raise ValueError("Inner OOF IDs or predicted labels are invalid")
        if part.groupby("prompt_id")["inner_fold"].nunique().max() != 1:
            raise ValueError("One prompt crosses inner folds")
        sizes = part.groupby("inner_fold")["prompt_id"].nunique().sort_index().tolist()
        if sizes != [32, 32, 32]:
            raise ValueError(f"Unexpected inner prompt counts for outer fold {outer}: {sizes}")
        confusion = confusion_matrix(part["true_model"], part["pred_model"], labels=LABELS)
        chosen, expected_pairs = select_pairs(confusion)
        actual_pairs = pair_table.loc[pair_table["outer_fold"].eq(outer)].reset_index(drop=True)
        for row, expected_row in zip(actual_pairs.to_dict("records"), expected_pairs):
            for key, value in expected_row.items():
                if row[key] != value:
                    raise ValueError(f"Pair count/selection differs in outer fold {outer}: {key}")
        selections[outer] = chosen
    return selections


def train(args: argparse.Namespace, dev: pd.DataFrame) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Training output directory is not empty: {args.output_dir}")
    selected = verified_selection(args, dev)
    started = time.perf_counter()
    seed_everything(SEED)
    labels = np.asarray([LABELS.index(label) for label in dev["model"]], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
    tokenized = tokenizer(dev["text"].tolist(), truncation=True, max_length=MAX_LENGTH, padding=False)
    encodings = {key: tokenized[key] for key in ("input_ids", "attention_mask")}
    device = torch.device("cuda:0")
    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    pretrained_revision = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "stage": "training",
        "model_id": MODEL_ID, "requested_revision": REVISION,
        "epochs": EPOCHS, "encoder_learning_rate": ENCODER_LR,
        "head_learning_rate": HEAD_LR, "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE, "seed": SEED,
        "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT, "pooling": "mean",
        "margin_weight": MARGIN_WEIGHT, "margin": MARGIN,
        "margin_loss": "mean relu(0.5 + partner_logit - true_logit) over selected-label batch rows; 0 if none",
        "selected_pairs": {str(fold): [[LABELS[a], LABELS[b]] for a, b in pairs]
                           for fold, pairs in selected.items()},
        "selection_config_sha256": sha256(args.selection_dir / "selection_config.json"),
        "inner_oof_sha256": sha256(args.selection_dir / "inner_oof_predictions.csv"),
        "pair_counts_sha256": sha256(args.selection_dir / "pair_counts.csv"),
        "n_rows_development": len(dev),
        "n_prompts_development": int(dev["prompt_id"].nunique()),
        "holdout_scored": False,
        "epoch_selection": "fixed final epoch",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **paths_and_hashes(args),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for outer in range(1, N_OUTER + 1):
        fold_started = time.perf_counter()
        fit_idx = np.flatnonzero(dev["fold"].to_numpy() != str(outer))
        valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(outer))
        partner = np.full(len(LABELS), -1, dtype=np.int64)
        for left, right in selected[outer]:
            partner[left] = right
            partner[right] = left
        partner = torch.tensor(partner, dtype=torch.long, device=device)
        seed_everything(SEED)
        model = MeanPoolClassifier(MODEL_ID, REVISION, DROPOUT, "mean").to(device)
        revision = getattr(model.encoder.config, "_commit_hash", None)
        if revision != REVISION:
            raise RuntimeError(f"Loaded RoBERTa revision {revision!r} differs from pinned SHA")
        if pretrained_revision is None:
            pretrained_revision = revision
        elif pretrained_revision != revision:
            raise RuntimeError("Pretrained revision changed between folds")
        counts = np.bincount(labels[fit_idx], minlength=len(LABELS))
        weights = len(fit_idx) / (len(LABELS) * counts.astype(np.float32))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        fit_loader = DataLoader(
            TokenizedDataset(encodings, labels, fit_idx), batch_size=BATCH_SIZE,
            shuffle=True, generator=torch.Generator().manual_seed(SEED),
            collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=True,
        )
        valid_loader = DataLoader(
            TokenizedDataset(encodings, labels, valid_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=True,
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
                    ce = F.cross_entropy(logits, y, weight=class_weights)
                    partners = partner[y]
                    eligible = partners.ge(0)
                    if eligible.any():
                        margin_loss = F.relu(
                            MARGIN + logits[eligible, partners[eligible]]
                            - logits[eligible, y[eligible]]
                        ).mean()
                        loss = ce + MARGIN_WEIGHT * margin_loss
                    else:
                        loss = ce
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
            print(f"outer={outer} epoch={epoch}/{EPOCHS} loss={epoch_losses[-1]:.6f}", flush=True)
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
        row = {
            "fold": outer,
            "train_rows": int(len(fit_idx)),
            "validation_rows": int(len(valid_idx)),
            "train_prompts": int(dev.iloc[fit_idx]["prompt_id"].nunique()),
            "validation_prompts": int(dev.iloc[valid_idx]["prompt_id"].nunique()),
            "selected_pairs": [[LABELS[a], LABELS[b]] for a, b in selected[outer]],
            "balanced_accuracy": float(balanced_accuracy_score(labels[valid_idx], predicted)),
            "family_accuracy": float(family_correct.mean()),
            "within_family_accuracy": float(
                (predicted == labels[valid_idx]).sum() / family_correct.sum()
            ),
            "epoch_train_loss": epoch_losses,
            "skipped_optimizer_steps": skipped,
            "duration_seconds": time.perf_counter() - fold_started,
        }
        fold_rows.append(row)
        if args.save_checkpoints:
            torch.save(model.cpu().state_dict(), args.output_dir / f"fold_{outer}_state_dict.pt")
        print(f"outer={outer} balanced_accuracy={row['balanced_accuracy']:.6f}", flush=True)
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    if np.isnan(scores).any():
        raise RuntimeError("Margin OOF scores are incomplete")
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
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    pd.DataFrame({
        "model": LABELS,
        "support": confusion.sum(axis=1),
        "correct": confusion.diagonal(),
        "recall": confusion.diagonal() / confusion.sum(axis=1),
    }).to_csv(args.output_dir / "class_metrics.csv", index=False)
    metrics = {
        **run_config,
        "experiment": "round3_targeted_within_family_margin",
        "pretrained_revision": pretrained_revision,
        "oof_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "oof_family_accuracy": float(family_correct.mean()),
        "oof_within_family_accuracy": float(
            (predicted == labels).sum() / family_correct.sum()
        ),
        "fold_metrics": fold_rows,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "nvidia_driver_version": driver_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "duration_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"OOF balanced_accuracy={metrics['oof_balanced_accuracy']:.6f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "train"), required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("outputs/e0_tfidf_svm/split_manifest.csv"))
    parser.add_argument("--e0-oof", type=Path, default=Path("outputs/e0_tfidf_svm/oof_predictions.csv"))
    parser.add_argument("--selection-dir", type=Path, default=Path("outputs/round3_margin_selection"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()
    try:
        dev = load_development(args.train, args.split_manifest, args.e0_oof)
        check_development(dev)
        if args.stage == "select":
            prepare_selection(args, dev)
        else:
            train(args, dev)
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
