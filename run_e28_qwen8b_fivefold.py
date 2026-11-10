#!/usr/bin/env python3
"""Locked E28 Qwen3-8B official-only five-fold classification experiment.

Preflight is CPU-only and writes the immutable experiment lock before any E28
validation scores are generated. Each GPU command trains one independent fold.
Summarize requires all five folds and compares a preregistered 1x addition to
the exact archived E24 fusion; no threshold or weight is selected on OOF data.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from run_e0 import LABELS, load_and_validate, sha256
from run_e28_qwen8b_lora_feasibility import (
    EXPECTED_SHA256, GRAD_ACCUM, LORA_TARGETS, LR, MAX_LENGTH,
    MODEL_REPO, MODEL_REVISION, PLANNED_EPOCHS, SEED, TRAIN_SHA256,
    WEIGHT_DECAY, make_base_model, package_version, verify_model_dir,
    verify_physical_gpu2_idle,
)
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "e28_qwen3_8b_fivefold.json"
SCRIPT_PATH = Path(__file__).resolve()
RESOURCE_SCRIPT_PATH = ROOT / "run_e28_qwen8b_lora_feasibility.py"
EXPECTED_E24_METRICS_SHA256 = "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557"
EXPECTED_E24_OOF_SHA256 = "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def config() -> dict:
    cfg = read_json(CONFIG_PATH)
    expected = {
        "experiment": "e28_qwen3_8b_official_only_fivefold",
        "protocol_version": 1,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "official_train_sha256": TRAIN_SHA256,
        "seed": SEED,
        "folds": [1, 2, 3, 4, 5],
        "max_length": MAX_LENGTH,
        "micro_batch_size": 1,
        "gradient_accumulation": GRAD_ACCUM,
        "epochs": PLANNED_EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_targets": LORA_TARGETS,
        "loss": "unweighted_cross_entropy",
        "evaluation": "final_epoch_only_no_epoch_or_threshold_selection",
        "fixed_fusion": "row_z_E24_E0c_plus_row_z_RoBERTa_plus_mean_three_row_z_ModernBERT_plus_1x_row_z_E28_logits",
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"E28 config changed from its resource-tested protocol: {key}")
    gate = cfg["promotion_gate"]
    if gate != {"reference": "fixed_E24_fusion", "fixed_added_branch_weight": 1.0,
                "min_balanced_accuracy_delta": 0.01,
                "min_non_decreasing_folds": 4,
                "max_allowed_class_recall_drop": 0.05,
                "paired_prompt_bootstrap_repeats": 20000}:
        raise ValueError("E28 promotion gate differs from preregistration")
    return cfg


def official_development(train: Path, e0_dir: Path, cfg: dict):
    if sha256(train) != cfg["official_train_sha256"]:
        raise ValueError("Official training CSV SHA-256 differs from preregistration")
    full = load_and_validate(train)
    dev, manifest, e0, folds, _ = load_e0_split(full, e0_dir, train)
    if len(full) != 2699 or full["prompt_id"].nunique() != 150:
        raise ValueError("Official training dimensions changed")
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Development partition dimensions changed")
    if len(manifest) != 150 or manifest["split"].eq("holdout").sum() != 30:
        raise ValueError("The reserved prompt holdout changed")
    for fold in cfg["folds"]:
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        train_frame, valid_frame = dev.iloc[train_idx], dev.iloc[valid_idx]
        if len(train_frame) != 1728 or len(valid_frame) != 432:
            raise ValueError(f"Fold {fold} row count changed")
        if train_frame["prompt_id"].nunique() != 96 or valid_frame["prompt_id"].nunique() != 24:
            raise ValueError(f"Fold {fold} prompt count changed")
        if set(train_frame["prompt_id"]) & set(valid_frame["prompt_id"]):
            raise ValueError(f"Fold {fold} leaks a prompt between fit and validation")
        if set(train_frame["model"]) != set(LABELS) or set(valid_frame["model"]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks an official target class")
    return dev, folds


def fixed_reference_paths(args) -> list[tuple[str, Path, str]]:
    e24_metrics = read_json(args.e24_oof.with_name("metrics.json"))
    if sha256(args.e24_oof.with_name("metrics.json")) != EXPECTED_E24_METRICS_SHA256:
        raise ValueError("E24 reference metrics changed")
    paths = [
        ("e24_oof", args.e24_oof, EXPECTED_E24_OOF_SHA256),
        ("e24_scores", args.e24_scores, e24_metrics["e24_scores_sha256"]),
        ("roberta_scores", args.roberta_scores, e24_metrics["roberta_scores_sha256"]),
        *[(f"modern_seed_{seed}", path, e24_metrics["modern_seed_score_sha256"][str(seed)])
          for seed, path in zip((42, 17, 73), args.modern_scores)],
    ]
    for name, path, expected in paths:
        if sha256(path) != expected:
            raise ValueError(f"Archived E24 reference changed: {name}")
    return paths


def lock_contents(args, cfg: dict) -> dict:
    fixed_reference_paths(args)
    return {
        "experiment": cfg["experiment"],
        "protocol_version": cfg["protocol_version"],
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "resource_script_sha256": sha256(RESOURCE_SCRIPT_PATH),
        "train_sha256": sha256(args.train),
        "e0_manifest_sha256": sha256(args.e0_dir / "split_manifest.csv"),
        "e0_oof_sha256": sha256(args.e0_dir / "oof_predictions.csv"),
        "e24_metrics_sha256": EXPECTED_E24_METRICS_SHA256,
        "e24_sources_sha256": {name: digest for name, _, digest in fixed_reference_paths(args)},
        "model_revision": cfg["model_revision"],
        "folds": cfg["folds"],
    }


def preflight(args) -> None:
    cfg = config()
    dev, _ = official_development(args.train, args.e0_dir, cfg)
    expected = lock_contents(args, cfg)
    path = args.output_dir / "protocol_lock.json"
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("Existing E28 protocol lock differs; use a new protocol version")
    else:
        write_new_json(path, expected)
    print(json.dumps({"stage": "cpu_preflight", "development_rows": len(dev),
                      "development_prompts": int(dev["prompt_id"].nunique()),
                      "protocol_lock_sha256": sha256(path),
                      "validation_accuracy_computed": False}, indent=2), flush=True)


def checked_lock(args, cfg: dict) -> dict:
    path = args.output_dir / "protocol_lock.json"
    lock = read_json(path)
    expected = lock_contents(args, cfg)
    if lock != expected:
        raise ValueError("E28 protocol inputs or code changed since CPU preflight")
    return lock


def checked_resource(args, cfg: dict) -> dict:
    report = read_json(args.resource_report)
    required = {
        "gate_passed": True,
        "model_repo": cfg["model_repo"],
        "model_revision": cfg["model_revision"],
        "train_sha256": cfg["official_train_sha256"],
        "model_sha256_verified": True,
        "validation_accuracy_computed": False,
    }
    for key, value in required.items():
        if report.get(key) != value:
            raise ValueError(f"E28 one-fold resource report failed or changed in {key}")
    return report


def checked_runtime(cfg: dict) -> dict:
    versions = {name: package_version(name) for name in
                ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
    expected = cfg["package_versions"]
    if (not (versions["torch"] or "").startswith(expected["torch_prefix"])
            or any(versions[name] != expected[name] for name in
                   ("transformers", "peft", "accelerate", "bitsandbytes"))):
        raise ValueError(f"E28 runtime package versions differ: {versions}")
    return versions


def gpu_preflight() -> str:
    import torch
    uuid = verify_physical_gpu2_idle()
    if torch.cuda.device_count() != 1 or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("E28 requires only the idle physical RTX 3090 GPU2")
    if torch.cuda.mem_get_info(0)[0] < 22 * 2**30:
        raise RuntimeError("E28 GPU2 has less than 22 GiB free")
    return uuid


def fit_fold(args) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    cfg = config()
    checked_lock(args, cfg)
    checked_resource(args, cfg)
    fold = args.fold
    if fold not in cfg["folds"]:
        raise ValueError("Requested fold is not in preregistration")
    fold_dir = args.output_dir / f"fold_{fold}"
    if fold_dir.exists():
        raise ValueError(f"E28 fold output already exists: {fold_dir}")
    versions = checked_runtime(cfg)
    sizes = verify_model_dir(args.model_dir)
    dev, folds = official_development(args.train, args.e0_dir, cfg)
    train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    train_frame, valid_frame = dev.iloc[train_idx], dev.iloc[valid_idx]
    uuid = gpu_preflight()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Pinned Qwen tokenizer lacks a pad token")
    label_ids = {name: index for index, name in enumerate(LABELS)}
    train_texts = train_frame["text"].tolist()
    train_targets = [label_ids[name] for name in train_frame["model"]]

    class Rows(Dataset):
        def __len__(self):
            return len(train_texts)

        def __getitem__(self, index):
            return train_texts[index], train_targets[index]

    def collate(batch):
        texts, targets = zip(*batch)
        output = tokenizer(list(texts), padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt", add_special_tokens=True)
        output["labels"] = torch.tensor(targets, dtype=torch.long)
        return output

    loader = DataLoader(Rows(), batch_size=1, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED),
                        num_workers=0, collate_fn=collate)
    torch.empty(1, device="cuda:0")
    torch.cuda.reset_peak_memory_stats(0)
    model = make_base_model(args.model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=LORA_TARGETS, modules_to_save=["score"], task_type="SEQ_CLS")
    model = get_peft_model(model, lora)
    if not any("score" in name and parameter.requires_grad
               for name, parameter in model.named_parameters()):
        raise ValueError("E28 18-class head is not trainable")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    planned_updates = updates_per_epoch * PLANNED_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(planned_updates * cfg["warmup_ratio"]),
        num_training_steps=planned_updates)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    completed, skipped, epoch_loss = 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, PLANNED_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {name: value.to("cuda:0") for name, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(**batch)
                loss = output.loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite E28 loss at fold {fold} epoch {epoch} step {step}")
            scaler.scale(loss).backward()
            total_loss += float(loss.item()) * GRAD_ACCUM
            if step % GRAD_ACCUM == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    completed += 1
                else:
                    skipped += 1
                optimizer.zero_grad(set_to_none=True)
        epoch_loss.append(total_loss / len(loader))
        print(f"E28 fold {fold} epoch {epoch}/{PLANNED_EPOCHS}: loss={epoch_loss[-1]:.6f}", flush=True)
    torch.cuda.synchronize(0)
    train_seconds = time.perf_counter() - started
    minimum_updates = math.ceil(planned_updates * 0.95)
    if completed < minimum_updates:
        raise FloatingPointError(f"Only {completed}/{planned_updates} E28 updates completed")
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    fold_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir = fold_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    # The previously resource-tested adapter reload is repeated before scoring.
    torch.cuda.reset_peak_memory_stats(0)
    model = make_base_model(args.model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    valid_texts = valid_frame["text"].tolist()
    scores = np.empty((len(valid_texts), len(LABELS)), dtype=np.float32)
    validation_started = time.perf_counter()
    with torch.inference_mode():
        for position, text in enumerate(valid_texts):
            encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt", add_special_tokens=True)
            encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
            logits = model(**encoded).logits
            if logits.shape != (1, len(LABELS)) or not torch.isfinite(logits).all():
                raise FloatingPointError(f"Invalid E28 logits at fold {fold}, row {position}")
            scores[position] = logits.float().cpu().numpy()[0]
    torch.cuda.synchronize(0)
    validation_seconds = time.perf_counter() - validation_started
    if not np.isfinite(scores).all():
        raise FloatingPointError("E28 OOF scores contain non-finite values")
    output = valid_frame.loc[:, ["id", "prompt_id", "model"]].rename(
        columns={"model": "true_model"}).copy()
    output["fold"] = str(fold)
    output["seed"] = SEED
    for index, label in enumerate(LABELS):
        output[f"score_{label}"] = scores[:, index]
    output["pred_model"] = np.asarray(LABELS)[scores.argmax(axis=1)]
    output.to_csv(fold_dir / "oof_predictions.csv", index=False)
    write_new_json(fold_dir / "run.json", {
        "experiment": cfg["experiment"], "fold": fold, "seed": SEED,
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "resource_report_sha256": sha256(args.resource_report),
        "model_revision": MODEL_REVISION, "model_required_file_sizes": sizes,
        "model_required_file_sha256": EXPECTED_SHA256,
        "packages": versions, "physical_gpu_uuid": uuid,
        "train_rows": len(train_idx), "validation_rows": len(valid_idx),
        "training_epochs_completed": PLANNED_EPOCHS,
        "epoch_loss": epoch_loss,
        "planned_optimizer_updates": planned_updates,
        "completed_optimizer_updates": completed,
        "skipped_optimizer_updates": skipped,
        "train_seconds": train_seconds,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "validation_inference_seconds": validation_seconds,
        "validation_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "validation_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "oof_predictions_sha256": sha256(fold_dir / "oof_predictions.csv"),
        "no_epoch_or_threshold_selection": True, "holdout_evaluated": False,
    })
    print(f"E28 fold {fold} complete; {len(output)} OOF rows saved", flush=True)


def aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} differs on OOF key {key}")


def summarize(args) -> None:
    cfg = config()
    checked_lock(args, cfg)
    checked_resource(args, cfg)
    dev, folds = official_development(args.train, args.e0_dir, cfg)
    pieces = []
    for fold in cfg["folds"]:
        folder = args.output_dir / f"fold_{fold}"
        path = folder / "oof_predictions.csv"
        run = read_json(folder / "run.json")
        if (run.get("experiment") != cfg["experiment"] or run.get("fold") != fold
                or run.get("seed") != SEED or run.get("training_epochs_completed") != PLANNED_EPOCHS
                or run.get("protocol_lock_sha256") != sha256(args.output_dir / "protocol_lock.json")
                or run.get("resource_report_sha256") != sha256(args.resource_report)
                or run.get("oof_predictions_sha256") != sha256(path)):
            raise ValueError(f"E28 fold {fold} run metadata or scores changed")
        frame = load_scores(path)
        if len(frame) != 432 or set(frame["fold"]) != {str(fold)} or set(frame["seed"]) != {SEED}:
            raise ValueError(f"E28 fold {fold} score row count, fold, or seed changed")
        pieces.append(frame)
    e28 = pd.concat(pieces, ignore_index=True).set_index("id", verify_integrity=True)
    if set(e28.index) != set(dev["id"]):
        raise ValueError("E28 five folds do not cover development rows exactly once")
    e28 = e28.loc[dev["id"]].reset_index()
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if not e28.loc[:, ["id", "prompt_id", "true_model"]].equals(expected):
        raise ValueError("E28 OOF identity differs from official development")
    if not np.array_equal(e28["fold"].to_numpy(dtype=int), folds):
        raise ValueError("E28 fold assignments differ from E0")
    fixed_reference_paths(args)
    e24 = load_scores(args.e24_scores)
    roberta = load_scores(args.roberta_scores)
    modern = [load_scores(path) for path in args.modern_scores]
    saved_e24 = pd.read_csv(args.e24_oof, dtype=str, keep_default_na=False)
    for name, frame in [("E24", e24), ("RoBERTa", roberta),
                        ("saved E24 fusion", saved_e24),
                        *[(f"ModernBERT seed {seed}", frame)
                          for seed, frame in zip((42, 17, 73), modern)]]:
        aligned(e28, frame, name)
    true = e28["true_model"].to_numpy()
    score_matrix = lambda frame: frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    baseline_scores = (row_standardize(score_matrix(e24))
                       + row_standardize(score_matrix(roberta))
                       + np.mean(np.stack([row_standardize(score_matrix(frame))
                                           for frame in modern]), axis=0))
    labels = np.asarray(LABELS)
    baseline = labels[baseline_scores.argmax(axis=1)]
    if not np.array_equal(baseline, saved_e24["pred_e16_with_e24"].to_numpy()):
        raise ValueError("Archived E24 fixed fusion did not reproduce exactly")
    if not np.isclose(balanced_accuracy_score(true, baseline), 0.425, atol=1e-12):
        raise ValueError("Archived E24 OOF balanced accuracy did not reproduce")
    candidate_scores = score_matrix(e28)
    single = labels[candidate_scores.argmax(axis=1)]
    added = labels[(baseline_scores + row_standardize(candidate_scores)).argmax(axis=1)]
    balanced = {
        "e24_fixed": float(balanced_accuracy_score(true, baseline)),
        "e28_single": float(balanced_accuracy_score(true, single)),
        "e24_plus_e28_fixed": float(balanced_accuracy_score(true, added)),
    }
    fold_rows = []
    for fold in cfg["folds"]:
        mask = folds == fold
        old = float(balanced_accuracy_score(true[mask], baseline[mask]))
        new = float(balanced_accuracy_score(true[mask], added[mask]))
        fold_rows.append({"fold": fold, "e24_fixed": old,
                          "e28_single": float(balanced_accuracy_score(true[mask], single[mask])),
                          "e24_plus_e28_fixed": new, "fusion_delta": new - old})
    class_rows = []
    for label in LABELS:
        mask = true == label
        old, new = float((baseline[mask] == label).mean()), float((added[mask] == label).mean())
        class_rows.append({"model": label, "support": int(mask.sum()),
                           "e24_fixed_recall": old,
                           "e28_single_recall": float((single[mask] == label).mean()),
                           "e24_plus_e28_fixed_recall": new, "fusion_delta": new - old})
    fold_frame, class_frame = pd.DataFrame(fold_rows), pd.DataFrame(class_rows)
    delta = balanced["e24_plus_e28_fixed"] - balanced["e24_fixed"]
    nondecrease = int((fold_frame["fusion_delta"] >= -1e-12).sum())
    worst = float(class_frame["fusion_delta"].min())
    gate = cfg["promotion_gate"]
    gates = {
        "delta_at_least_0_01": delta >= gate["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": nondecrease >= gate["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": worst > -gate["max_allowed_class_recall_drop"] + 1e-12,
    }
    report = {
        "experiment": cfg["experiment"], "stage": "five_fold_oof_analysis",
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "resource_report_sha256": sha256(args.resource_report),
        "e24_reference_oof_sha256": sha256(args.e24_oof),
        "rows": len(e28), "prompts": int(e28["prompt_id"].nunique()),
        "labels": list(LABELS), "balanced_accuracy": balanced,
        "e28_single_confusion_matrix": confusion_matrix(true, single, labels=LABELS).tolist(),
        "paired_prompt_bootstrap_e24_plus_e28_minus_e24": paired_prompt_bootstrap(
            true, e28["prompt_id"].to_numpy(), baseline, added,
            repeats=gate["paired_prompt_bootstrap_repeats"], seed=SEED),
        "fusion_wrong_to_right": int(((baseline != true) & (added == true)).sum()),
        "fusion_right_to_wrong": int(((baseline == true) & (added != true)).sum()),
        "folds_non_decreasing": nondecrease,
        "worst_class_recall_delta": worst, "fusion_delta": delta,
        "gate": gates, "gate_passed_against_e24": all(gates.values()),
        "comparison_reference_must_be_rechecked_at_promotion": True,
        "holdout_evaluated": False, "fusion_weight_tuned": False,
    }
    analysis = args.output_dir / "analysis"
    if analysis.exists():
        raise ValueError("E28 analysis output already exists")
    analysis.mkdir(parents=True, exist_ok=False)
    e28.to_csv(analysis / "oof_predictions.csv", index=False)
    fold_frame.to_csv(analysis / "fold_metrics.csv", index=False)
    class_frame.to_csv(analysis / "class_metrics.csv", index=False)
    write_new_json(analysis / "metrics.json", report)
    print(json.dumps({"balanced_accuracy": balanced, "gate": gates}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("preflight", "fit-fold", "summarize"):
        stage = stages.add_parser(name)
        stage.add_argument("--train", type=Path, default=Path("data/train.csv"))
        stage.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
        stage.add_argument("--output-dir", type=Path, default=Path("outputs/e28_qwen8b_fivefold"))
        stage.add_argument("--resource-report", type=Path,
                           default=Path("outputs/e28_qwen8b_resource_mirror/resource_report.json"))
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
        if name == "fit-fold":
            stage.add_argument("--fold", type=int, required=True, choices=range(1, 6))
            stage.add_argument("--model-dir", type=Path, default=Path("data/models/qwen3-8b-b968826d"))
    args = parser.parse_args()
    {"preflight": preflight, "fit-fold": fit_fold, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
