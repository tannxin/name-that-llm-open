#!/usr/bin/env python3
"""Locked E27 ModernBERT head-first experiment; feasibility then five-fold OOF."""

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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_e0 import LABELS, N_FOLDS, sha256
from run_e25_external_modernbert import (
    BATCH_SIZE, DROPOUT, ENCODER_LR, HEAD_LR, MAX_LENGTH, MODEL_ID,
    RESOURCE_LIMIT_GIB, RESOURCE_LIMIT_SECONDS_PER_RECIPE, REVISION,
    TOTAL_LIMIT_SECONDS, WARMUP_RATIO, WEIGHT_DECAY, evaluate, locked_input,
    make_loader,
)
from run_finetune_encoder import MeanPoolClassifier, seed_everything


HEAD_EPOCHS = 2
TOP_EPOCHS = 10
TOP_BLOCKS = 2


def set_trainable(model: MeanPoolClassifier, stage_name: str) -> dict[str, int]:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if stage_name == "head_only":
        pass
    elif stage_name == "top_two":
        if not hasattr(model.encoder, "layers") or not hasattr(model.encoder, "final_norm"):
            raise RuntimeError("Pinned ModernBERT encoder layer structure changed")
        if len(model.encoder.layers) <= TOP_BLOCKS:
            raise RuntimeError("ModernBERT has fewer layers than preregistered")
        for layer in model.encoder.layers[-TOP_BLOCKS:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for parameter in model.encoder.final_norm.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(stage_name)
    head = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    return {"trainable_head_parameters": head, "trainable_encoder_parameters": encoder}


def set_training_modes(model: MeanPoolClassifier, stage_name: str) -> None:
    model.train()
    model.encoder.eval()
    if stage_name == "top_two":
        for layer in model.encoder.layers[-TOP_BLOCKS:]:
            layer.train()
        model.encoder.final_norm.train()


def train_stage(model, loader, epochs, stage_name, device):
    trainable = set_trainable(model, stage_name)
    parameter_groups = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad],
         "lr": ENCODER_LR},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad],
         "lr": HEAD_LR},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses = []
    skipped_steps = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        set_training_modes(model, stage_name)
        weighted_loss_sum = 0.0
        seen = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids, attention_mask)
                per_row = F.cross_entropy(logits.float(), labels, reduction="none")
                loss = (per_row * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {stage_name}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            else:
                skipped_steps += 1
            weighted_loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
        mean_loss = weighted_loss_sum / seen
        losses.append(mean_loss)
        print(f"stage={stage_name} epoch={epoch}/{epochs} train_loss={mean_loss:.6f}",
              flush=True)
    torch.cuda.synchronize()
    return {
        "stage": stage_name, "epochs": epochs, "rows": len(loader.dataset),
        "optimizer_steps_planned": total_steps,
        "skipped_optimizer_steps": skipped_steps,
        "epoch_train_loss": losses,
        "seconds": time.perf_counter() - started,
        **trainable,
    }


def fit_fold(args, fold, dev, external, tokenized, labels, device):
    started = time.perf_counter()
    seed_everything(args.seed)
    model = MeanPoolClassifier(MODEL_ID, REVISION, DROPOUT, "mean").to(device)
    torch.cuda.reset_peak_memory_stats(0)
    commit = getattr(model.encoder.config, "_commit_hash", None)
    if commit != REVISION:
        raise RuntimeError(f"Loaded revision {commit} differs from locked {REVISION}")
    n_dev = len(dev)
    fit_idx = np.flatnonzero(dev["fold"].to_numpy() != str(fold))
    valid_idx = np.flatnonzero(dev["fold"].to_numpy() == str(fold))
    if len(fit_idx) != 1728 or len(valid_idx) != 432:
        raise ValueError("Official fold sizes changed")
    if set(dev.iloc[fit_idx]["prompt_id"]) & set(dev.iloc[valid_idx]["prompt_id"]):
        raise ValueError("Official training and validation prompt overlap")
    weights = np.ones(n_dev + len(external), dtype=np.float32)
    if args.recipe == "X":
        weights[np.flatnonzero(dev["model"].eq("qwen3.8-max").to_numpy())] = 1.25
        weights[n_dev:] = external["sample_weight"].to_numpy(dtype=np.float32)
        train_idx = np.concatenate((fit_idx, np.arange(n_dev, len(labels))))
        by_class = np.bincount(labels[train_idx], weights=weights[train_idx],
                               minlength=len(LABELS))
        if not np.allclose(by_class, 120.0, atol=1e-4):
            raise ValueError("External mixture effective class weights changed")
    else:
        train_idx = fit_idx
        by_class = np.bincount(labels[train_idx], weights=weights[train_idx],
                               minlength=len(LABELS))
        if not np.allclose(by_class, 96.0, atol=1e-4):
            raise ValueError("Official effective class weights changed")
    stages = []
    for stage_name, epochs in (("head_only", HEAD_EPOCHS),
                               ("top_two", TOP_EPOCHS)):
        train_loader = make_loader(args.tokenizer, tokenized, labels, weights,
                                   train_idx, shuffle=True, seed=args.seed)
        stages.append(train_stage(model, train_loader, epochs, stage_name, device))
    valid_loader = make_loader(args.tokenizer, tokenized, labels, weights,
                               valid_idx, shuffle=False, seed=args.seed)
    eval_info, scores = evaluate(model, valid_loader, n_dev, device,
                                 feasibility=args.stage == "feasibility")
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated(0) / 2**30
    duration = time.perf_counter() - started
    result = {
        "fold": fold, "recipe": args.recipe, "seed": args.seed,
        "official_train_rows": len(fit_idx), "official_valid_rows": len(valid_idx),
        "external_train_rows": len(external) if args.recipe == "X" else 0,
        "effective_training_weight_by_class": {
            name: float(by_class[j]) for j, name in enumerate(LABELS)},
        "stages": stages, "inference": eval_info,
        "gpu_peak_allocated_gib": peak_gib, "duration_seconds": duration,
        "pretrained_revision": commit, "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.stage == "feasibility":
        result["resource_gate_passed"] = bool(
            peak_gib <= RESOURCE_LIMIT_GIB
            and duration <= RESOURCE_LIMIT_SECONDS_PER_RECIPE)
    del model
    torch.cuda.empty_cache()
    return result, scores


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.seed != 42:
        raise ValueError("E27 preregistered seed is 42 only")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")
    dev, external, meta = locked_input(args)
    labels = np.asarray([
        LABELS.index(name)
        for name in pd.concat((dev["model"], external["model"]), ignore_index=True)
    ], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
    texts = pd.concat((dev["text"], external["text"]), ignore_index=True).tolist()
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
    tokenized = {name: tokenized[name] for name in ("input_ids", "attention_mask")}
    args.tokenizer = tokenizer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "e27_headfirst_modernbert", "stage": args.stage,
        "recipe": args.recipe, "seed": args.seed,
        "model_id": MODEL_ID, "requested_revision": REVISION,
        "head_epochs": HEAD_EPOCHS, "top_epochs": TOP_EPOCHS,
        "top_encoder_blocks": TOP_BLOCKS,
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
        "dropout": DROPOUT, "pooling": "mean",
        "official_train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "e0_oof_sha256": sha256(args.e0_oof),
        "selected_external_sha256": sha256(args.e24_selected_external),
        "external_selection": meta,
        "source_protocol_uses_official_global_length_statistics": True,
        "validation_selection": "none; fixed final epoch",
        "resource_limits": {
            "peak_gpu_gib": RESOURCE_LIMIT_GIB,
            "single_fold_minutes": RESOURCE_LIMIT_SECONDS_PER_RECIPE / 60,
            "five_fold_hours": TOTAL_LIMIT_SECONDS / 3600,
        },
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
        result, scores = fit_fold(args, fold, dev, external, tokenized, labels, device)
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
            raise RuntimeError("Five-fold time cap exceeded; incomplete OOF not scored")
    if not np.isfinite(full_scores).all():
        raise RuntimeError("Incomplete five-fold OOF logits")
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
        "experiment": "e27_headfirst_modernbert", "recipe": args.recipe,
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
    parser.add_argument("--recipe", required=True, choices=("O", "X"))
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
