#!/usr/bin/env python3
"""Locked E25 external-data ModernBERT experiment (feasibility or five-fold fit)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_e0 import LABELS, N_FOLDS, sha256
from run_e24_prvsiyan_external_ablation import select_external
from run_finetune_encoder import MeanPoolClassifier, load_development, seed_everything


MODEL_ID = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SPLIT_SHA = "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15"
E0_OOF_SHA = "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa"
PROMPT_SELECTION_SHA = "2b917303a82218ba36a11cf61288276d72cf7d8d19f0feb914eb1434673ed266"
SELECTED_EXTERNAL_SHA = "d7f8b7c9b0480ec896777e667589bf7d2fe12e503aa8f867c0708d0656ad6191"
MAX_LENGTH = 256
BATCH_SIZE = 16
ENCODER_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.1
RESOURCE_LIMIT_GIB = 22.0
RESOURCE_LIMIT_SECONDS_PER_RECIPE = 25 * 60
TOTAL_LIMIT_SECONDS = 3 * 3600


class WeightedDataset(Dataset):
    def __init__(self, encodings, labels, weights, indices):
        self.encodings = encodings
        self.labels = labels
        self.weights = weights
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": int(self.labels[index]),
            "weight": float(self.weights[index]),
            "row_index": index,
        }


def collate(tokenizer, rows):
    result = tokenizer.pad(
        [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in rows],
        padding=True, return_tensors="pt",
    )
    result["labels"] = torch.tensor([x["labels"] for x in rows], dtype=torch.long)
    result["weight"] = torch.tensor([x["weight"] for x in rows], dtype=torch.float32)
    result["row_index"] = torch.tensor([x["row_index"] for x in rows], dtype=torch.long)
    return result


def make_loader(tokenizer, encodings, labels, weights, indices, *, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        WeightedDataset(encodings, labels, weights, indices), batch_size=BATCH_SIZE,
        shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True,
        collate_fn=lambda rows: collate(tokenizer, rows),
    )


def locked_input(args):
    for path, expected in (
        (args.train, TRAIN_SHA), (args.split_manifest, SPLIT_SHA),
        (args.e0_oof, E0_OOF_SHA),
        (args.e24_selected_prompts, PROMPT_SELECTION_SHA),
        (args.e24_selected_external, SELECTED_EXTERNAL_SHA),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Locked input SHA differs: {path}")
    external, prompt_manifest, meta = select_external(
        args.external_source, args.audit_manifest, args.audit_report,
    )
    if len(external) != 3620 or external["model"].nunique() != 17:
        raise ValueError("E24 external sample count/class coverage differs")
    if meta["selected_rows_by_class"].get("qwen3.8-max") != 0:
        raise ValueError("Unverified Qwen rows are present")
    saved_prompt = pd.read_csv(args.e24_selected_prompts, dtype=str, keep_default_na=False)
    candidate_prompt = prompt_manifest.astype(str)
    if not saved_prompt.equals(candidate_prompt):
        raise ValueError("E24 selected prompt manifest changed")
    saved_external = pd.read_csv(args.e24_selected_external, dtype=str, keep_default_na=False)
    keys = ["id", "prompt_id", "model", "served_model_id", "served_provider", "provider_route"]
    if not saved_external.loc[:, keys].equals(external.loc[:, keys].astype(str)):
        raise ValueError("E24 selected external rows or routes changed")
    dev = load_development(args.train, args.split_manifest, args.e0_oof)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Official development partition changed")
    if set(dev["prompt_id"]) & set(external["prompt_id"]):
        raise ValueError("External and official prompt IDs overlap")
    return dev, external, meta


def train_stage(model, loader, epochs, stage_name, device):
    groups = [
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},
        {"params": model.classifier.parameters(), "lr": HEAD_LR},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses = []
    skipped = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        numerator, denominator = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids, mask)
                per_row = F.cross_entropy(logits.float(), labels, reduction="none")
                loss = (per_row * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {stage_name}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            else:
                skipped += 1
            numerator += float(loss.item()) * len(labels)
            denominator += len(labels)
        value = numerator / denominator
        losses.append(value)
        print(f"{stage_name} epoch={epoch}/{epochs} train_loss={value:.6f}", flush=True)
    torch.cuda.synchronize()
    return {
        "stage": stage_name, "epochs": epochs, "rows": len(loader.dataset),
        "optimizer_steps_planned": total_steps, "skipped_optimizer_steps": skipped,
        "epoch_train_loss": losses, "seconds": time.perf_counter() - started,
    }


def evaluate(model, loader, n_dev, device, *, feasibility):
    model.eval()
    if feasibility:
        batches = 2
        started = time.perf_counter()
        n = 0
        with torch.inference_mode():
            for j, batch in enumerate(loader):
                if j >= batches:
                    break
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(batch["input_ids"].to(device),
                                   batch["attention_mask"].to(device))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite validation forward logits")
                n += len(batch["labels"])
        torch.cuda.synchronize()
        return {"inference_rows": n, "inference_seconds": time.perf_counter() - started}, None
    scores = np.full((n_dev, len(LABELS)), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device))
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite official validation logits")
            scores[batch["row_index"].numpy()] = values
    return {"inference_rows": len(loader.dataset)}, scores


def fit_fold(args, fold, dev, external, tokenized, labels, meta, device):
    start = time.perf_counter()
    seed_everything(args.seed)
    model = MeanPoolClassifier(MODEL_ID, REVISION, DROPOUT, "mean").to(device)
    # Initialize the CUDA context before resetting peak stats; this Torch build
    # rejects the reset call while the context is still lazy/uninitialized.
    torch.cuda.reset_peak_memory_stats(0)
    commit = getattr(model.encoder.config, "_commit_hash", None)
    if commit != REVISION:
        raise RuntimeError(f"Loaded revision {commit} differs from locked {REVISION}")
    n_dev = len(dev)
    fit_idx = np.flatnonzero(dev["fold"].to_numpy() != str(fold))
    valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(fold))
    if len(fit_idx) != 1728 or len(valid_idx) != 432:
        raise ValueError(f"Fold {fold} changed official train/validation size")
    if set(dev.iloc[fit_idx]["prompt_id"]) & set(dev.iloc[valid_idx]["prompt_id"]):
        raise ValueError(f"Fold {fold} has overlapping official prompt IDs")
    ext_idx = np.arange(n_dev, n_dev + len(external), dtype=np.int64)
    official_weights = np.ones(n_dev, dtype=np.float32)
    if args.recipe == "B":
        official_weights[dev["model"].eq("qwen3.8-max").to_numpy()] = 1.25
        ext_weights = external["sample_weight"].to_numpy(dtype=np.float32)
        stage_specs = [("mixed", 12, np.concatenate((fit_idx, ext_idx)),
                        np.concatenate((official_weights, ext_weights)))]
    else:
        counts = external["model"].value_counts().to_dict()
        ext_weights = np.asarray([250.0 / counts[name] for name in external["model"]],
                                 dtype=np.float32)
        stage_specs = [
            ("external", 3, ext_idx,
             np.concatenate((official_weights, ext_weights))),
            ("official_adaptation", 12, fit_idx,
             np.concatenate((official_weights, ext_weights))),
        ]
    stages = []
    for stage_name, epochs, indices, weights in stage_specs:
        by_class = np.bincount(labels[indices], weights=weights[indices],
                               minlength=len(LABELS))
        if stage_name == "mixed" and not np.allclose(by_class, 120.0, atol=1e-4):
            raise ValueError("Mixed 18-class effective weights are not all 120")
        if stage_name == "external" and not (np.allclose(by_class[:14], 250) and
                                              np.isclose(by_class[14], 0) and
                                              np.allclose(by_class[15:], 250)):
            raise ValueError("External stage 17-class weights changed")
        if stage_name == "official_adaptation" and not np.allclose(by_class, 96):
            raise ValueError("Official adaptation class weights changed")
        loader = make_loader(args.tokenizer, tokenized, labels, weights, indices,
                             shuffle=True, seed=args.seed)
        stage = train_stage(model, loader, epochs, f"fold={fold}:{stage_name}", device)
        stage["effective_weight_by_class"] = {
            name: float(by_class[j]) for j, name in enumerate(LABELS)
        }
        stages.append(stage)
    valid_weights = np.ones(n_dev + len(external), dtype=np.float32)
    valid_loader = make_loader(args.tokenizer, tokenized, labels, valid_weights, valid_idx,
                               shuffle=False, seed=args.seed)
    eval_info, scores = evaluate(model, valid_loader, n_dev, device,
                                 feasibility=args.stage == "feasibility")
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated(0) / 2**30
    duration = time.perf_counter() - start
    result = {
        "fold": fold, "recipe": args.recipe, "seed": args.seed,
        "official_train_rows": len(fit_idx), "official_valid_rows": len(valid_idx),
        "external_rows": len(external), "stages": stages,
        "inference": eval_info, "gpu_peak_allocated_gib": peak_gib,
        "duration_seconds": duration, "pretrained_revision": commit,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.stage == "feasibility":
        result["resource_gate_passed"] = bool(
            peak_gib <= RESOURCE_LIMIT_GIB and duration <= RESOURCE_LIMIT_SECONDS_PER_RECIPE
        )
    else:
        result["balanced_accuracy"] = float(balanced_accuracy_score(
            dev.iloc[valid_idx]["model"], np.asarray(LABELS)[scores[valid_idx].argmax(axis=1)]))
    del model
    torch.cuda.empty_cache()
    return result, scores


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.stage == "fit" and args.seed not in (42, 17):
        raise ValueError("Formal E25 seeds are restricted to 42 and conditional 17")
    if args.stage == "feasibility" and args.seed != 42:
        raise ValueError("Feasibility is seed 42 only")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")
    dev, external, meta = locked_input(args)
    labels = np.asarray([LABELS.index(x) for x in
                         pd.concat((dev["model"], external["model"]), ignore_index=True)],
                        dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
    texts = pd.concat((dev["text"], external["text"]), ignore_index=True).tolist()
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
    tokenized = {name: tokenized[name] for name in ("input_ids", "attention_mask")}
    args.tokenizer = tokenizer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "e25_external_modernbert", "stage": args.stage,
        "recipe": args.recipe, "seed": args.seed, "model_id": MODEL_ID,
        "requested_revision": REVISION, "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE, "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR, "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO, "dropout": DROPOUT,
        "pooling": "mean", "official_train_sha256": TRAIN_SHA,
        "split_manifest_sha256": SPLIT_SHA, "e0_oof_sha256": E0_OOF_SHA,
        "external_selection": meta, "selected_prompt_manifest_sha256": PROMPT_SELECTION_SHA,
        "selected_external_sha256": SELECTED_EXTERNAL_SHA,
        "source_protocol_uses_official_global_length_statistics": True,
        "validation_selection": "none; fixed final epoch",
        "resource_limits": {"peak_gpu_gib": RESOURCE_LIMIT_GIB,
                            "single_fold_recipe_minutes": 25,
                            "five_fold_recipe_hours": 3},
        "torch_version": torch.__version__,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    device = torch.device("cuda:0")
    folds = (1,) if args.stage == "feasibility" else range(1, N_FOLDS + 1)
    full_scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    results = []
    started = time.perf_counter()
    for fold in folds:
        result, scores = fit_fold(args, fold, dev, external, tokenized, labels, meta, device)
        results.append(result)
        (args.output_dir / "fold_resource.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"fold={fold} recipe={args.recipe} seconds={result['duration_seconds']:.1f} "
              f"peak_gib={result['gpu_peak_allocated_gib']:.2f}", flush=True)
        if args.stage == "feasibility":
            print(f"resource_gate_passed={result['resource_gate_passed']}", flush=True)
            return
        valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(fold))
        full_scores[valid_idx] = scores[valid_idx]
        if time.perf_counter() - started > TOTAL_LIMIT_SECONDS:
            raise RuntimeError("Five-fold resource time cap exceeded; incomplete OOF not scored")
    if not np.isfinite(full_scores).all():
        raise RuntimeError("Five-fold OOF logits incomplete")
    predictions = np.asarray(LABELS)[full_scores.argmax(axis=1)]
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = dev["model"]
    oof["pred_model"] = predictions
    oof["fold"] = dev["fold"]
    oof["correct"] = predictions == dev["model"].to_numpy()
    for j, name in enumerate(LABELS):
        oof[f"score_{name}"] = full_scores[:, j]
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    metrics = {
        "experiment": "e25_external_modernbert", "recipe": args.recipe,
        "seed": args.seed, "pretrained_revision": REVISION,
        "oof_balanced_accuracy": float(balanced_accuracy_score(dev["model"], predictions)),
        "fold_metrics": results,
        "gpu_total_duration_seconds": time.perf_counter() - started,
        "source_protocol_uses_official_global_length_statistics": True,
        "holdout_evaluated": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OOF BA={metrics['oof_balanced_accuracy']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("feasibility", "fit"))
    parser.add_argument("--recipe", required=True, choices=("B", "C"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("outputs/e0_tfidf_svm/split_manifest.csv"))
    parser.add_argument("--e0-oof", type=Path,
                        default=Path("outputs/e0_tfidf_svm/oof_predictions.csv"))
    parser.add_argument("--external-source", type=Path,
                        default=Path("data/external/prvsiyan/passages.csv"))
    parser.add_argument("--audit-manifest", type=Path,
                        default=Path("data/external/prvsiyan/e24_accepted_manifest.csv"))
    parser.add_argument("--audit-report", type=Path,
                        default=Path("outputs/e24_prvsiyan_audit/report.json"))
    parser.add_argument("--e24-selected-prompts", type=Path,
                        default=Path("outputs/e24_prvsiyan_fit/selected_prompt_ids.csv"))
    parser.add_argument("--e24-selected-external", type=Path,
                        default=Path("outputs/e24_prvsiyan_fit/selected_external.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
