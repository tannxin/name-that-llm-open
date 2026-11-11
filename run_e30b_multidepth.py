#!/usr/bin/env python3
"""E30-B: preregistered three-depth readout on the E30-A official-only backbone.

``preflight`` is CPU-only and writes a new immutable protocol lock. GPU stages
require the lock and a successful resource test. This script never downloads
weights and never reads the E0 holdout or hidden-test answers for evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import run_e30_supervised_embedding8b as a
from run_e0 import LABELS, N_FOLDS, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize


CONFIG_PATH = Path(__file__).with_name("configs") / "e30b_qwen3_embedding8b_multidepth.json"
SCRIPT_PATH = Path(__file__).resolve()
PROTOCOL_VERSION = 1
A_SCRIPT_SHA256 = "82c99f7211cbbd7cc0dc5e64b5bac568a25855e42d55c08078cf6a76056ce375"
A_CONFIG_SHA256 = "06d2806fcc86fe2573bc4525494859de2d171ac94ba98319430b5da8bf0e084f"
A_OOF_SHA256 = "0feb265eb77dab60206db3ad1c7ca469724d32a52562567efa284af8532d266d"
READOUT_INDICES = (12, 24, 36)
EXEMPT_A_CONFIG_KEYS = {"experiment", "pooling", "readout", "promotion_gate"}


def read_json(path: Path) -> dict:
    return a.read_json(path)


def write_new_json(path: Path, payload: dict) -> None:
    a.write_new_json(path, payload)


def config() -> dict:
    if sha256(a.SCRIPT_PATH) != A_SCRIPT_SHA256 or sha256(a.CONFIG_PATH) != A_CONFIG_SHA256:
        raise ValueError("E30-A reusable code or config differs from the pinned version")
    prior = a.config()
    cfg = read_json(CONFIG_PATH)
    if cfg.get("experiment") != "e30_b_qwen3_embedding8b_multidepth_official_only":
        raise ValueError("Wrong E30-B experiment")
    if set(cfg) != set(prior) | {"readout"}:
        raise ValueError("E30-B config keys differ from E30-A plus readout")
    for key in prior:
        if key not in EXEMPT_A_CONFIG_KEYS and cfg[key] != prior[key]:
            raise ValueError(f"E30-B changed non-readout E30-A setting {key}")
    if cfg["pooling"] != "last_non_padding_token_from_hidden_states_12_24_36":
        raise ValueError("E30-B pooling changed")
    if cfg["readout"] != {
        "hidden_state_indices": [12, 24, 36],
        "hidden_state_index_zero_is_embedding": True,
        "classifier": "concatenate_three_4096_dim_vectors_then_one_linear_12288_to_18",
        "normalization": "none",
        "learned_layer_weights": False,
    }:
        raise ValueError("E30-B readout architecture changed")
    if cfg["folds"] != list(range(1, N_FOLDS + 1)):
        raise ValueError("E30-B folds changed")
    expected_gate = dict(prior["promotion_gate"])
    expected_gate["reference"] = "fixed_E24_plus_E30_A_fusion"
    if cfg["promotion_gate"] != expected_gate:
        raise ValueError("E30-B comparison rule differs from the preregistration")
    return cfg


def a_lock(args) -> dict:
    prior_args = SimpleNamespace(train=args.train, e0_dir=args.e0_dir,
                                 output_dir=args.a_output_dir)
    return a.lock_input(prior_args, a.config())


def lock_input(args, cfg: dict) -> dict:
    lock_path = args.output_dir / "protocol_lock.json"
    lock = read_json(lock_path)
    a_lock(args)
    expected = {
        "experiment": cfg["experiment"],
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "a_config_sha256": sha256(a.CONFIG_PATH),
        "a_script_sha256": sha256(a.SCRIPT_PATH),
        "a_protocol_lock_sha256": sha256(args.a_output_dir / "protocol_lock.json"),
        "train_sha256": sha256(args.train),
        "e0_manifest_sha256": sha256(args.e0_dir / "split_manifest.csv"),
        "e0_oof_sha256": sha256(args.e0_dir / "oof_predictions.csv"),
        "model_revision": cfg["model_revision"],
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"E30-B protocol lock differs in {key}")
    return lock


def cpu_readout_contract(cfg: dict) -> None:
    """Check indexing and concatenation without model weights or validation scores."""
    layers = tuple(cfg["readout"]["hidden_state_indices"])
    assert layers == READOUT_INDICES
    # Position 0 is the embedding, so 36 is the final transformer output.
    states = tuple(np.full((2, 4, 3), index, dtype=np.float32)
                   for index in range(37))
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int64)
    rows = np.arange(2)
    positions = mask.sum(axis=1) - 1
    features = np.concatenate([states[layer][rows, positions]
                               for layer in layers], axis=-1)
    if features.shape != (2, 9):
        raise ValueError("E30-B readout feature shape changed")
    if not np.array_equal(features[0], np.repeat([12, 24, 36], 3)):
        raise ValueError("E30-B readout layer order changed")


def preflight(args) -> None:
    cfg = config()
    if (args.output_dir / "protocol_lock.json").exists():
        raise ValueError("E30-B protocol lock already exists; refusing to overwrite")
    a_lock(args)
    _, dev, _, folds = a.official_development(args.train, args.e0_dir, cfg)
    cpu_readout_contract(cfg)
    report = {
        "experiment": cfg["experiment"],
        "stage": "protocol_lock_before_e30b_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "a_config_sha256": sha256(a.CONFIG_PATH),
        "a_script_sha256": sha256(a.SCRIPT_PATH),
        "a_protocol_lock_sha256": sha256(args.a_output_dir / "protocol_lock.json"),
        "train_sha256": sha256(args.train),
        "e0_manifest_sha256": sha256(args.e0_dir / "split_manifest.csv"),
        "e0_oof_sha256": sha256(args.e0_dir / "oof_predictions.csv"),
        "model_repo": cfg["model_repo"],
        "model_revision": cfg["model_revision"],
        "readout_indices": list(READOUT_INDICES),
        "readout_cpu_contract_passed": True,
        "development_rows": len(dev),
        "development_prompts": int(dev["prompt_id"].nunique()),
        "fold_rows": {str(f): int((folds == f).sum()) for f in cfg["folds"]},
        "validation_accuracy_computed": False,
        "b_model_loaded": False,
        "gpu_used": False,
    }
    write_new_json(args.output_dir / "protocol_lock.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_model(model_dir: Path, cfg: dict, *, adapter_dir: Path | None = None):
    import torch

    # Reuse the exact E30-A quantization, LoRA targets and adapter loading.
    a_model = a.build_model(model_dir, cfg, adapter_dir=adapter_dir)
    encoder = a_model.encoder
    del a_model

    class MultiDepthClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Linear(
                4096 * len(READOUT_INDICES), len(LABELS),
                dtype=torch.float32, device="cuda:0")

        def forward(self, input_ids, attention_mask):
            output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True)
            states = output.hidden_states
            if states is None or len(states) != 37:
                raise ValueError("Pinned 36-layer embedding model must return 37 hidden states")
            positions = attention_mask.long().sum(dim=1) - 1
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)
            features = torch.cat(
                [states[index][rows, positions] for index in READOUT_INDICES], dim=-1)
            if features.shape[-1] != 12288:
                raise ValueError("E30-B concatenated readout width changed")
            return self.head(features.float())

    return MultiDepthClassifier()


def reload_model(model_dir: Path, fold_dir: Path, cfg: dict):
    from safetensors.torch import load_file

    model = build_model(model_dir, cfg, adapter_dir=fold_dir / "adapter")
    model.head.load_state_dict(load_file(str(fold_dir / "head.safetensors")), strict=True)
    return model


def prepare_gpu(args):
    cfg = config()
    lock_input(args, cfg)
    versions = a.runtime_versions(cfg)
    model_hashes = a.verify_model_dir(args.model_dir, cfg)
    _, dev, _, folds = a.official_development(args.train, args.e0_dir, cfg)
    tokenizer = a.load_tokenizer(args.model_dir)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    a.gpu_preflight(cfg)
    import torch
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    return cfg, versions, model_hashes, dev, folds, tokenizer


def resource(args) -> None:
    import torch

    if (args.output_dir / "resource_report.json").exists():
        raise ValueError("E30-B resource report already exists")
    cfg, versions, hashes, dev, folds, tokenizer = prepare_gpu(args)
    train_frame = dev.iloc[np.flatnonzero(folds != 1)]
    label_to_int = {label: index for index, label in enumerate(LABELS)}
    loader = a.make_loader(train_frame["text"].tolist(),
                           [label_to_int[x] for x in train_frame["model"]],
                           tokenizer, cfg, shuffle=True)
    a.reset_cuda_peak_stats()
    model = build_model(args.model_dir, cfg)
    train_report = a.train(model, loader, cfg, epochs=1)
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    resource_dir = args.output_dir / "resource_one_epoch"
    resource_dir.mkdir(parents=True, exist_ok=False)
    a.save_adapter_head(model, tokenizer, resource_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    a.reset_cuda_peak_stats()
    reload_start = time.perf_counter()
    model = reload_model(args.model_dir, resource_dir, cfg)
    reload_seconds = time.perf_counter() - reload_start
    source = train_frame["text"].tolist()
    texts = [source[index % len(source)] for index in range(4000)]
    inference_loader = a.make_loader(texts, None, tokenizer, cfg, shuffle=False)
    start = time.perf_counter()
    scores = a.predict(model, inference_loader)
    inference_seconds = time.perf_counter() - start
    inference_peak_allocated = torch.cuda.max_memory_allocated(0)
    inference_peak_reserved = torch.cuda.max_memory_reserved(0)
    if scores.shape != (4000, len(LABELS)):
        raise ValueError("E30-B resource inference row count changed")
    limits = cfg["resource_gate"]
    gate = {
        "train_peak_under_limit": train_peak_reserved < limits["max_train_peak_gib"] * 2**30,
        "inference_peak_under_limit": inference_peak_reserved < limits["max_train_peak_gib"] * 2**30,
        "one_epoch_under_limit": train_report["train_seconds"] <= limits["max_one_epoch_seconds"],
        "projected_three_epochs_under_limit": train_report["train_seconds"] * 3 <= limits["max_projected_three_epoch_seconds"],
        "four_thousand_inference_under_limit": inference_seconds <= limits["max_four_thousand_inference_seconds"],
        "optimizer_updates_under_skip_limit": (
            train_report["completed_optimizer_updates"]
            >= math.ceil(train_report["planned_optimizer_updates"]
                         * limits["min_completed_optimizer_update_fraction"])),
        "finite_four_thousand_logits": bool(np.isfinite(scores).all()),
    }
    report = {
        "experiment": cfg["experiment"], "stage": "one_fold_resource_only",
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "model_revision": cfg["model_revision"], "model_file_sha256": hashes,
        "packages": versions, "fold_training_side": 1,
        "training_rows": len(train_frame), "training_epochs": 1,
        **train_report,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "adapter_reload_seconds": reload_seconds,
        "unlabeled_inference_rows": len(texts),
        "inference_seconds": inference_seconds,
        "inference_peak_allocated_bytes": inference_peak_allocated,
        "inference_peak_reserved_bytes": inference_peak_reserved,
        "projected_five_fold_training_seconds": train_report["train_seconds"] * 15,
        "deployment_model_count": cfg["deployment_model_count"],
        "hidden_test_runtime_proven": False,
        "validation_accuracy_computed": False,
        "gate": gate, "gate_passed": all(gate.values()),
    }
    write_new_json(args.output_dir / "resource_report.json", report)
    print(json.dumps({"gate": gate, "gate_passed": report["gate_passed"]}, indent=2))
    if not report["gate_passed"]:
        raise RuntimeError("E30-B resource gate failed; do not train five folds")


def checked_resource(args, cfg: dict) -> dict:
    report = read_json(args.output_dir / "resource_report.json")
    if (report.get("experiment") != cfg["experiment"]
            or report.get("config_sha256") != sha256(CONFIG_PATH)
            or report.get("script_sha256") != sha256(SCRIPT_PATH)
            or report.get("protocol_lock_sha256") != sha256(args.output_dir / "protocol_lock.json")
            or report.get("model_revision") != cfg["model_revision"]
            or not report.get("gate_passed")):
        raise ValueError("E30-B resource gate is missing, failed, or stale")
    return report


def fit_fold(args) -> None:
    import torch

    cfg = config()
    lock_input(args, cfg)
    checked_resource(args, cfg)
    if args.fold not in cfg["folds"]:
        raise ValueError("Requested fold is outside the locked folds")
    fold_dir = args.output_dir / f"fold_{args.fold}"
    if fold_dir.exists():
        raise ValueError("E30-B fold output already exists")
    cfg, versions, hashes, dev, folds, tokenizer = prepare_gpu(args)
    train_index = np.flatnonzero(folds != args.fold)
    valid_index = np.flatnonzero(folds == args.fold)
    label_to_int = {label: index for index, label in enumerate(LABELS)}
    fit = dev.iloc[train_index]
    loader = a.make_loader(fit["text"].tolist(),
                           [label_to_int[x] for x in fit["model"]],
                           tokenizer, cfg, shuffle=True)
    a.reset_cuda_peak_stats()
    model = build_model(args.model_dir, cfg)
    train_report = a.train(model, loader, cfg, epochs=cfg["epochs"])
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    fold_dir.mkdir(parents=True, exist_ok=False)
    a.save_adapter_head(model, tokenizer, fold_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    a.reset_cuda_peak_stats()
    model = reload_model(args.model_dir, fold_dir, cfg)
    valid = dev.iloc[valid_index]
    valid_loader = a.make_loader(valid["text"].tolist(), None, tokenizer,
                                 cfg, shuffle=False)
    start = time.perf_counter()
    scores = a.predict(model, valid_loader)
    inference_seconds = time.perf_counter() - start
    if scores.shape != (len(valid_index), len(LABELS)):
        raise ValueError("E30-B validation score shape changed")
    output = valid.loc[:, ["id", "prompt_id", "model"]].rename(
        columns={"model": "true_model"}).copy()
    output["fold"] = str(args.fold)
    for index, label in enumerate(LABELS):
        output[f"score_{label}"] = scores[:, index]
    output["pred_model"] = np.asarray(LABELS)[scores.argmax(axis=1)]
    output.to_csv(fold_dir / "oof_predictions.csv", index=False)
    write_new_json(fold_dir / "run.json", {
        "experiment": cfg["experiment"], "fold": args.fold,
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "resource_report_sha256": sha256(args.output_dir / "resource_report.json"),
        "model_revision": cfg["model_revision"], "model_file_sha256": hashes,
        "packages": versions, "train_rows": len(train_index),
        "validation_rows": len(valid_index), **train_report,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "validation_inference_seconds": inference_seconds,
        "validation_inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "validation_inference_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "oof_predictions_sha256": sha256(fold_dir / "oof_predictions.csv"),
        "no_epoch_or_threshold_selection": True,
        "holdout_evaluated": False,
    })
    print(f"E30-B fold {args.fold} complete; no promotion decision")


def aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} differs in {key}")


def summarize(args) -> None:
    cfg = config()
    lock_input(args, cfg)
    checked_resource(args, cfg)
    _, dev, _, folds = a.official_development(args.train, args.e0_dir, cfg)
    pieces = []
    for fold in cfg["folds"]:
        folder = args.output_dir / f"fold_{fold}"
        run = read_json(folder / "run.json")
        path = folder / "oof_predictions.csv"
        for key, expected in (
            ("experiment", cfg["experiment"]), ("fold", fold),
            ("config_sha256", sha256(CONFIG_PATH)),
            ("script_sha256", sha256(SCRIPT_PATH)),
            ("protocol_lock_sha256", sha256(args.output_dir / "protocol_lock.json")),
            ("resource_report_sha256", sha256(args.output_dir / "resource_report.json")),
            ("oof_predictions_sha256", sha256(path)),
        ):
            if run.get(key) != expected:
                raise ValueError(f"E30-B fold {fold} run differs in {key}")
        if (run.get("epochs") != 3 or run.get("train_rows") != 1728
                or run.get("validation_rows") != 432
                or run.get("completed_optimizer_updates", 0) < 0.95 * run.get("planned_optimizer_updates", 1)
                or not run.get("no_epoch_or_threshold_selection")
                or run.get("holdout_evaluated") is not False):
            raise ValueError(f"E30-B fold {fold} training protocol changed")
        frame = load_scores(path)
        if len(frame) != 432 or set(frame["fold"]) != {str(fold)}:
            raise ValueError(f"E30-B fold {fold} OOF dimensions changed")
        pieces.append(frame)
    b = pd.concat(pieces, ignore_index=True).set_index("id", verify_integrity=True)
    if set(b.index) != set(dev["id"]):
        raise ValueError("E30-B fivefold OOF fails exact development coverage")
    b = b.loc[dev["id"]].reset_index()
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(
        columns={"model": "true_model"})
    if not b.loc[:, ["id", "prompt_id", "true_model"]].equals(expected):
        raise ValueError("E30-B OOF identities differ from official development")
    if not np.array_equal(b["fold"].to_numpy(dtype=int), folds):
        raise ValueError("E30-B fold assignment differs from E0")
    b_scores = b.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(b_scores).all():
        raise ValueError("E30-B OOF logits are nonfinite")
    labels = np.asarray(LABELS)
    if not np.array_equal(b["pred_model"].to_numpy(), labels[b_scores.argmax(axis=1)]):
        raise ValueError("E30-B saved predictions differ from argmax")

    a_oof = args.a_output_dir / "analysis" / "oof_predictions.csv"
    if sha256(a_oof) != A_OOF_SHA256:
        raise ValueError("E30-A OOF differs from its independent audit")
    prior = load_scores(a_oof)
    aligned(b, prior, "E30-A")
    a_scores = prior.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(a_scores).all():
        raise ValueError("E30-A logits are nonfinite")

    e24_metrics_path = args.e24_oof.with_name("metrics.json")
    if sha256(e24_metrics_path) != a.EXPECTED_E24_METRICS_SHA256:
        raise ValueError("E24 metrics differ from locked audit")
    if sha256(args.e24_oof) != a.EXPECTED_E24_OOF_SHA256:
        raise ValueError("E24 OOF differs from locked audit")
    e24_metrics = read_json(e24_metrics_path)
    sources = [
        ("E24 E0c", args.e24_scores, e24_metrics["e24_scores_sha256"]),
        ("RoBERTa", args.roberta_scores, e24_metrics["roberta_scores_sha256"]),
        *[(f"ModernBERT seed {seed}", path,
           e24_metrics["modern_seed_score_sha256"][str(seed)])
          for seed, path in zip((42, 17, 73), args.modern_scores)],
    ]
    for name, path, expected_hash in sources:
        if sha256(path) != expected_hash:
            raise ValueError(f"{name} score file differs from E24 lock")
    e24 = load_scores(args.e24_scores)
    roberta = load_scores(args.roberta_scores)
    modern = [load_scores(path) for path in args.modern_scores]
    for name, frame in [("E24", e24), ("RoBERTa", roberta),
                        *[(f"ModernBERT seed {seed}", frame)
                          for seed, frame in zip((42, 17, 73), modern)]]:
        aligned(b, frame, name)
    baseline_scores = (
        row_standardize(e24.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
        + row_standardize(roberta.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
        + np.mean(np.stack([
            row_standardize(frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
            for frame in modern]), axis=0))
    saved_e24 = pd.read_csv(args.e24_oof, dtype=str, keep_default_na=False)
    aligned(b, saved_e24, "Archived E24")
    e24_pred = labels[baseline_scores.argmax(axis=1)]
    if not np.array_equal(e24_pred, saved_e24["pred_e16_with_e24"].to_numpy()):
        raise ValueError("E24 fixed fusion does not reproduce archived predictions")

    true = b["true_model"].to_numpy()
    weight = cfg["promotion_gate"]["fixed_added_branch_weight"]
    a_pred = labels[(baseline_scores + weight * row_standardize(a_scores)).argmax(axis=1)]
    b_pred = labels[(baseline_scores + weight * row_standardize(b_scores)).argmax(axis=1)]
    a_single = labels[a_scores.argmax(axis=1)]
    b_single = labels[b_scores.argmax(axis=1)]
    scores = {
        "e24_fixed": float(balanced_accuracy_score(true, e24_pred)),
        "e30_a_single": float(balanced_accuracy_score(true, a_single)),
        "e30_b_single": float(balanced_accuracy_score(true, b_single)),
        "e24_plus_e30_a_fixed": float(balanced_accuracy_score(true, a_pred)),
        "e24_plus_e30_b_fixed": float(balanced_accuracy_score(true, b_pred)),
    }
    if abs(scores["e24_fixed"] - 0.425) > 1e-12 or abs(scores["e24_plus_e30_a_fixed"] - 0.4486111111111111) > 1e-12:
        raise ValueError("Archived E24/E30-A comparator BA did not reproduce")
    fold_rows, class_rows = [], []
    for fold in cfg["folds"]:
        mask = folds == fold
        old = float(balanced_accuracy_score(true[mask], a_pred[mask]))
        new = float(balanced_accuracy_score(true[mask], b_pred[mask]))
        fold_rows.append({
            "fold": fold,
            "e30_a_single": float(balanced_accuracy_score(true[mask], a_single[mask])),
            "e30_b_single": float(balanced_accuracy_score(true[mask], b_single[mask])),
            "e24_plus_e30_a_fixed": old,
            "e24_plus_e30_b_fixed": new,
            "b_minus_a_fusion_delta": new - old,
        })
    for label in LABELS:
        mask = true == label
        old = float((a_pred[mask] == label).mean())
        new = float((b_pred[mask] == label).mean())
        class_rows.append({
            "model": label, "support": int(mask.sum()),
            "e30_a_single_recall": float((a_single[mask] == label).mean()),
            "e30_b_single_recall": float((b_single[mask] == label).mean()),
            "e24_plus_e30_a_fixed_recall": old,
            "e24_plus_e30_b_fixed_recall": new,
            "b_minus_a_fusion_delta": new - old,
        })
    fold_frame = pd.DataFrame(fold_rows)
    class_frame = pd.DataFrame(class_rows)
    limits = cfg["promotion_gate"]
    delta = scores["e24_plus_e30_b_fixed"] - scores["e24_plus_e30_a_fixed"]
    nondecrease = int((fold_frame["b_minus_a_fusion_delta"] >= -1e-12).sum())
    worst_class = float(class_frame["b_minus_a_fusion_delta"].min())
    gate = {
        "delta_at_least_0_01": delta >= limits["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": nondecrease >= limits["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": worst_class > -limits["max_allowed_class_recall_drop"] + 1e-12,
    }
    report = {
        "experiment": cfg["experiment"], "stage": "fivefold_oof_analysis",
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "protocol_lock_sha256": sha256(args.output_dir / "protocol_lock.json"),
        "a_oof_sha256": sha256(a_oof),
        "e24_reference_oof_sha256": sha256(args.e24_oof),
        "rows": len(b), "prompts": int(b["prompt_id"].nunique()),
        "labels": list(LABELS), "balanced_accuracy": scores,
        "e30_b_single_confusion_matrix": confusion_matrix(true, b_single, labels=LABELS).tolist(),
        "paired_prompt_bootstrap_b_minus_a_fixed": paired_prompt_bootstrap(
            true, b["prompt_id"].to_numpy(), a_pred, b_pred,
            repeats=limits["paired_prompt_bootstrap_repeats"], seed=cfg["seed"]),
        "b_fusion_wrong_to_right": int(((a_pred != true) & (b_pred == true)).sum()),
        "b_fusion_right_to_wrong": int(((a_pred == true) & (b_pred != true)).sum()),
        "folds_non_decreasing": nondecrease,
        "worst_class_recall_delta": worst_class,
        "fusion_delta_vs_a": delta,
        "gate": gate, "gate_passed_vs_a": all(gate.values()),
        "holdout_evaluated": False, "fusion_weight_tuned": False,
        "deployment_and_kaggle_proven": False,
    }
    analysis_dir = args.output_dir / "analysis"
    if analysis_dir.exists():
        raise ValueError("E30-B analysis already exists; refusing to overwrite")
    analysis_dir.mkdir(parents=True, exist_ok=False)
    b.to_csv(analysis_dir / "oof_predictions.csv", index=False)
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
        stage.add_argument("--a-output-dir", type=Path,
                           default=Path("outputs/e30_qwen3_embedding8b_v3"))
        stage.add_argument("--output-dir", type=Path,
                           default=Path("outputs/e30b_qwen3_embedding8b_v1"))
        if name in ("resource", "fit-fold"):
            stage.add_argument("--model-dir", type=Path, required=True,
                               help="Verified local Qwen3-Embedding-8B snapshot")
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
            TimeoutError) as exc:
        print(f"E30-B {args.stage} failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
