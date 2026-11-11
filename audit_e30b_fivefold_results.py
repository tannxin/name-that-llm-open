#!/usr/bin/env python3
"""Independently audit the locked E30-B five-fold development OOF.

Copy only the small E30-B JSON/CSV artifacts to the Git-ignored snapshot
directory after all five folds and ``summarize`` have completed. This CPU-only
audit reads the archived development labels and score files, but does not read
holdout answers, hidden test data, or model weights. It writes only a
public-safe aggregate JSON after every check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = ROOT / "outputs/e30b_qwen3_embedding8b_audit_snapshot"
DEFAULT_OUTPUT = ROOT / "docs/e30b_multidepth_audit_aggregate.json"
A_SNAPSHOT = ROOT / "outputs/e30_qwen3_embedding8b_audit_snapshot"
KEYS = ["id", "prompt_id", "true_model", "fold"]
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash",
    "kimi-k3",
]
SCORE_COLS = [f"score_{label}" for label in LABELS]
EXPECTED = {
    "b_script": "d660aa5e70cb25066cdecdd1e9425e8328cbdc68b655753e6f75443433f8895b",
    "b_config": "b46f82add551e7440c4fc2dd4ae9fccfec5aa9ea627df0fc2fbe68d7e4ce5d09",
    "a_script": "82c99f7211cbbd7cc0dc5e64b5bac568a25855e42d55c08078cf6a76056ce375",
    "a_config": "06d2806fcc86fe2573bc4525494859de2d171ac94ba98319430b5da8bf0e084f",
    "a_lock": "7d6abc27b606dad9247e9cd0668c27fd5efe5af291f537bce6dcfd72279731cf",
    "a_resource": "5ce64b2b557b13f4503193e27ae67f35b23e3506c73be060e0d479f35e7a9368",
    "a_metrics": "77cba0bdb60d7b720493bc54517c8cf3825ceff4698060567642fa52e4425065",
    "a_oof": "0feb265eb77dab60206db3ad1c7ca469724d32a52562567efa284af8532d266d",
    "a_audit": "4094c41f15703b1a94d64ebcec245832b3783e9622b738ceb4199c0c5df345ec",
    "e28_audit": "da593d5f335fb137fc88a1b35aea670edaa63cce9496e34e22caeca8f6c2e3dc",
    "e24_metrics": "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557",
    "e24_oof": "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448",
    "train": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
    "e0_manifest": "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15",
    "e0_oof": "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fixed_hash(path: Path, value: str) -> None:
    require(digest(path) == EXPECTED[value], f"Immutable {value} hash differs: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str) -> None:
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) < 1e-12,
            f"{label} differs: {actual!r} vs {expected!r}")


def row_z(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True, ddof=0)
    require(np.isfinite(scale).all() and (scale > 0).all(), "Non-finite or flat score row")
    return centered / scale


def balanced(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean([(predicted[true == label] == label).mean() for label in LABELS]))


def read_scores(path: Path, reference: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
    require(len(frame) == 2160 and frame.id.is_unique, f"Wrong row count or duplicate ID: {path}")
    require(set(frame.id) == set(reference.id), f"ID coverage differs: {path}")
    require(all(col in frame for col in [*KEYS, *SCORE_COLS]), f"Missing score columns: {path}")
    frame = frame.set_index("id").loc[reference.id].reset_index()
    require(frame[KEYS].equals(reference), f"Row identity or order differs: {path}")
    scores = frame[SCORE_COLS].to_numpy(dtype=np.float64)
    require(scores.shape == (2160, 18) and np.isfinite(scores).all(), f"Bad logits: {path}")
    return frame, scores


def compare_csv(path: Path, expected: list[dict]) -> None:
    saved = pd.read_csv(path).to_dict(orient="records")
    require(len(saved) == len(expected), f"CSV row count differs: {path}")
    for actual, wanted in zip(saved, expected):
        require(set(actual) == set(wanted), f"CSV columns differ: {path}")
        for key, value in wanted.items():
            if isinstance(value, (int, float)):
                close(actual[key], value, f"{path} {key}")
            else:
                require(actual[key] == value, f"{path} {key} differs")


def gate_vs(old_score: float, new_score: float, old_folds: list[float],
            new_folds: list[float], old_classes: list[float],
            new_classes: list[float], limits: dict) -> dict:
    deltas = [new - old for old, new in zip(old_folds, new_folds)]
    class_deltas = [new - old for old, new in zip(old_classes, new_classes)]
    return {
        "delta_at_least_0_01": new_score - old_score >= limits["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": sum(x >= -1e-12 for x in deltas)
        >= limits["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": min(class_deltas)
        > -limits["max_allowed_class_recall_drop"] + 1e-12,
    }


def comparison(old_name: str, old_score: float, old_folds: list[float],
               old_classes: list[float], new_score: float, new_folds: list[float],
               new_classes: list[float], limits: dict) -> dict:
    fold_deltas = [new - old for old, new in zip(old_folds, new_folds)]
    class_deltas = [new - old for old, new in zip(old_classes, new_classes)]
    gate = gate_vs(old_score, new_score, old_folds, new_folds, old_classes, new_classes, limits)
    return {
        "reference": old_name,
        "balanced_accuracy_delta": new_score - old_score,
        "fold_deltas": fold_deltas,
        "class_recall_deltas": dict(zip(LABELS, class_deltas)),
        "worst_class": LABELS[int(np.argmin(class_deltas))],
        "worst_class_recall_delta": min(class_deltas),
        "folds_non_decreasing": sum(x >= -1e-12 for x in fold_deltas),
        "gate": gate, "gate_passed": all(gate.values()),
    }


def main(snapshot: Path, output: Path) -> None:
    # Refuse partial data before touching any validation scores or output file.
    required = [snapshot / "protocol_lock.json", snapshot / "resource_report.json",
                snapshot / "analysis/oof_predictions.csv", snapshot / "analysis/metrics.json",
                snapshot / "analysis/fold_metrics.csv", snapshot / "analysis/class_metrics.csv"]
    for fold in range(1, 6):
        required += [snapshot / f"fold_{fold}/oof_predictions.csv",
                     snapshot / f"fold_{fold}/run.json"]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "E30-B fivefold snapshot is incomplete: " + ", ".join(missing))
    require(not output.exists(), f"Refusing to overwrite existing audit: {output}")

    b_cfg_path = ROOT / "configs/e30b_qwen3_embedding8b_multidepth.json"
    a_cfg_path = ROOT / "configs/e30_qwen3_embedding8b.json"
    fixed_hash(ROOT / "run_e30b_multidepth.py", "b_script")
    fixed_hash(b_cfg_path, "b_config")
    fixed_hash(ROOT / "run_e30_supervised_embedding8b.py", "a_script")
    fixed_hash(a_cfg_path, "a_config")
    fixed_hash(A_SNAPSHOT / "protocol_lock.json", "a_lock")
    fixed_hash(A_SNAPSHOT / "resource_report.json", "a_resource")
    fixed_hash(A_SNAPSHOT / "analysis/metrics.json", "a_metrics")
    fixed_hash(A_SNAPSHOT / "analysis/oof_predictions.csv", "a_oof")
    fixed_hash(ROOT / "docs/e30_embedding8b_fivefold_audit_aggregate.json", "a_audit")
    fixed_hash(ROOT / "docs/e28_qwen8b_fivefold_audit_aggregate.json", "e28_audit")
    fixed_hash(ROOT / "outputs/e24_prvsiyan_analysis/metrics.json", "e24_metrics")
    fixed_hash(ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv", "e24_oof")
    manifest_path = ROOT / "outputs/e0_tfidf_svm/split_manifest.csv"
    e0_path = ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv"
    fixed_hash(manifest_path, "e0_manifest")
    fixed_hash(e0_path, "e0_oof")

    cfg = read_json(b_cfg_path)
    a_cfg = read_json(a_cfg_path)
    limits = cfg["promotion_gate"]
    require(cfg["experiment"] == "e30_b_qwen3_embedding8b_multidepth_official_only", "Wrong experiment")
    require(cfg["model_repo"] == a_cfg["model_repo"] == "Qwen/Qwen3-Embedding-8B", "Wrong model")
    require(cfg["model_revision"] == a_cfg["model_revision"], "Model revision changed")
    require(cfg["readout"]["hidden_state_indices"] == [12, 24, 36], "Readout layers changed")
    require(cfg["folds"] == [1, 2, 3, 4, 5], "Fold sequence changed")
    require(set(cfg) == set(a_cfg) | {"readout"}, "B config differs beyond the new readout key")
    for key in a_cfg:
        if key not in {"experiment", "pooling", "readout", "promotion_gate"}:
            require(cfg[key] == a_cfg[key], f"B changed non-readout setting {key}")
    expected_gate = dict(a_cfg["promotion_gate"])
    expected_gate["reference"] = "fixed_E24_plus_E30_A_fusion"
    require(limits == expected_gate, "B promotion gate differs from A beyond comparator")
    require(limits["reference"] == "fixed_E24_plus_E30_A_fusion"
            and limits["fixed_added_branch_weight"] == 1.0
            and limits["paired_prompt_bootstrap_repeats"] == 20_000, "Comparison protocol changed")

    lock_path = snapshot / "protocol_lock.json"
    resource_path = snapshot / "resource_report.json"
    lock, resource = read_json(lock_path), read_json(resource_path)
    require(lock["experiment"] == cfg["experiment"] and lock["protocol_version"] == 1
            and lock["stage"] == "protocol_lock_before_e30b_validation", "Invalid B protocol lock")
    for key, wanted in {
        "config_sha256": EXPECTED["b_config"], "script_sha256": EXPECTED["b_script"],
        "a_config_sha256": EXPECTED["a_config"], "a_script_sha256": EXPECTED["a_script"],
        "a_protocol_lock_sha256": EXPECTED["a_lock"], "train_sha256": EXPECTED["train"],
        "e0_manifest_sha256": EXPECTED["e0_manifest"], "e0_oof_sha256": EXPECTED["e0_oof"],
        "model_repo": cfg["model_repo"], "model_revision": cfg["model_revision"],
        "readout_indices": [12, 24, 36], "development_rows": 2160,
        "development_prompts": 120,
        "fold_rows": {str(fold): 432 for fold in range(1, 6)},
        "readout_cpu_contract_passed": True, "validation_accuracy_computed": False,
        "b_model_loaded": False, "gpu_used": False,
    }.items():
        require(lock.get(key) == wanted, f"B lock differs in {key}")

    a_resource = read_json(A_SNAPSHOT / "resource_report.json")
    require(resource["experiment"] == cfg["experiment"]
            and resource["stage"] == "one_fold_resource_only", "Invalid B resource stage")
    for key, wanted in {
        "config_sha256": EXPECTED["b_config"], "script_sha256": EXPECTED["b_script"],
        "protocol_lock_sha256": digest(lock_path), "model_revision": cfg["model_revision"],
        "model_file_sha256": a_resource["model_file_sha256"],
        "fold_training_side": 1, "training_rows": 1728, "training_epochs": 1,
        "unlabeled_inference_rows": 4000, "deployment_model_count": 1,
        "validation_accuracy_computed": False, "hidden_test_runtime_proven": False,
    }.items():
        require(resource.get(key) == wanted, f"B resource differs in {key}")
    packages = resource["packages"]
    for key in ("transformers", "peft", "accelerate", "bitsandbytes"):
        require(packages[key] == cfg["package_versions"][key], f"Package differs: {key}")
    require(packages["torch"].startswith(cfg["package_versions"]["torch_prefix"]), "PyTorch differs")
    require(resource["epochs"] == 1 and len(resource["epoch_loss"]) == 1
            and np.isfinite(resource["epoch_loss"]).all(), "Resource training epoch differs")
    require(resource["completed_optimizer_updates"] + resource["skipped_optimizer_updates"]
            == resource["planned_optimizer_updates"] == 216, "Resource optimizer updates differ")
    gib = 2**30
    resource_gate = {
        "train_peak_under_limit": resource["train_peak_reserved_bytes"] < cfg["resource_gate"]["max_train_peak_gib"] * gib,
        "inference_peak_under_limit": resource["inference_peak_reserved_bytes"] < cfg["resource_gate"]["max_train_peak_gib"] * gib,
        "one_epoch_under_limit": resource["train_seconds"] <= cfg["resource_gate"]["max_one_epoch_seconds"],
        "projected_three_epochs_under_limit": resource["train_seconds"] * 3 <= cfg["resource_gate"]["max_projected_three_epoch_seconds"],
        "four_thousand_inference_under_limit": resource["inference_seconds"] <= cfg["resource_gate"]["max_four_thousand_inference_seconds"],
        "optimizer_updates_under_skip_limit": resource["completed_optimizer_updates"]
        >= math.ceil(resource["planned_optimizer_updates"]
                     * cfg["resource_gate"]["min_completed_optimizer_update_fraction"]),
        "finite_four_thousand_logits": resource["gate"]["finite_four_thousand_logits"],
    }
    require(resource["gate"] == resource_gate and all(resource_gate.values())
            and resource["gate_passed"], "B resource gate is not independently reproducible")
    close(resource["projected_five_fold_training_seconds"], resource["train_seconds"] * 15,
          "Projected five-fold training time")

    e0 = pd.read_csv(e0_path, dtype=str, keep_default_na=False)
    require(len(e0) == 2160 and e0.id.is_unique and e0.prompt_id.nunique() == 120,
            "E0 development identity is invalid")
    require(set(e0.true_model) == set(LABELS) and e0.true_model.value_counts().eq(120).all(),
            "E0 label balance changed")
    require(e0.prompt_id.value_counts().eq(18).all()
            and e0.groupby("prompt_id").true_model.nunique().eq(18).all()
            and e0.groupby("prompt_id").fold.nunique().eq(1).all(), "E0 prompt structure changed")
    require(e0.fold.value_counts().eq(432).all(), "E0 fold size changed")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    dev_prompts = set(manifest.loc[manifest.split == "development", "prompt_id"])
    require(len(dev_prompts) == 120 and set(e0.prompt_id) == dev_prompts
            and manifest.split.eq("holdout").sum() == 30, "E0 holdout partition changed")
    reference = e0[KEYS].copy()

    pieces, fold_resources = [], []
    for fold in range(1, 6):
        folder = snapshot / f"fold_{fold}"
        path = folder / "oof_predictions.csv"
        run = read_json(folder / "run.json")
        for key, wanted in {
            "experiment": cfg["experiment"], "fold": fold,
            "config_sha256": EXPECTED["b_config"], "script_sha256": EXPECTED["b_script"],
            "protocol_lock_sha256": digest(lock_path),
            "resource_report_sha256": digest(resource_path),
            "model_revision": cfg["model_revision"],
            "model_file_sha256": a_resource["model_file_sha256"],
            "packages": packages, "train_rows": 1728, "validation_rows": 432,
            "epochs": 3, "oof_predictions_sha256": digest(path),
            "no_epoch_or_threshold_selection": True, "holdout_evaluated": False,
        }.items():
            require(run.get(key) == wanted, f"Fold {fold} run differs in {key}")
        require(len(run["epoch_loss"]) == 3 and np.isfinite(run["epoch_loss"]).all(),
                f"Fold {fold} epoch losses invalid")
        require(run["completed_optimizer_updates"] + run["skipped_optimizer_updates"]
                == run["planned_optimizer_updates"] == 648, f"Fold {fold} optimizer updates invalid")
        require(run["completed_optimizer_updates"] >= math.ceil(648 * 0.95),
                f"Fold {fold} skipped too many optimizer updates")
        require(0 < run["train_seconds"] <= cfg["resource_gate"]["max_projected_three_epoch_seconds"],
                f"Fold {fold} train time exceeds resource budget")
        require(0 < run["train_peak_reserved_bytes"] < cfg["resource_gate"]["max_train_peak_gib"] * gib
                and 0 < run["validation_inference_peak_reserved_bytes"] < cfg["resource_gate"]["max_train_peak_gib"] * gib
                and run["validation_inference_seconds"] > 0, f"Fold {fold} peak memory or time invalid")
        piece = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
        require(len(piece) == 432 and piece.id.is_unique and set(piece.fold) == {str(fold)},
                f"Fold {fold} OOF dimensions invalid")
        wanted = e0.loc[e0.fold == str(fold), KEYS]
        require(set(piece.id) == set(wanted.id), f"Fold {fold} IDs differ from E0")
        ordered = piece.set_index("id").loc[wanted.id].reset_index()
        require(ordered[KEYS].equals(wanted.reset_index(drop=True)),
                f"Fold {fold} row/prompt/label/fold alignment differs from E0")
        require(piece.prompt_id.nunique() == 24
                and set(piece.prompt_id).isdisjoint(set(e0.loc[e0.fold != str(fold), "prompt_id"])),
                f"Fold {fold} prompt leakage")
        require(all(col in piece for col in [*SCORE_COLS, "pred_model"]),
                f"Fold {fold} missing logits/prediction")
        logits = piece[SCORE_COLS].to_numpy(dtype=np.float64)
        require(logits.shape == (432, 18) and np.isfinite(logits).all(),
                f"Fold {fold} logits invalid")
        require(np.array_equal(piece.pred_model.to_numpy(), np.asarray(LABELS)[logits.argmax(axis=1)]),
                f"Fold {fold} saved predictions differ from logits")
        pieces.append(piece)
        fold_resources.append({
            "fold": fold, "train_seconds": run["train_seconds"],
            "train_peak_reserved_bytes": run["train_peak_reserved_bytes"],
            "validation_seconds": run["validation_inference_seconds"],
            "validation_peak_reserved_bytes": run["validation_inference_peak_reserved_bytes"],
            "completed_updates": run["completed_optimizer_updates"],
            "planned_updates": run["planned_optimizer_updates"],
        })

    joined = pd.concat(pieces, ignore_index=True)
    require(joined.id.is_unique and set(joined.id) == set(reference.id), "Five folds do not cover E0")
    joined = joined.set_index("id").loc[reference.id].reset_index()
    summary_path = snapshot / "analysis/oof_predictions.csv"
    saved = pd.read_csv(summary_path, dtype={key: str for key in KEYS}, keep_default_na=False)
    require(saved.equals(joined), "B summarized OOF differs from the five fold files")
    require(saved[KEYS].equals(reference), "B summarized identities differ from E0")
    b_logits = saved[SCORE_COLS].to_numpy(dtype=np.float64)
    require(np.isfinite(b_logits).all(), "B summary logits are non-finite")
    label_array = np.asarray(LABELS)
    true = reference.true_model.to_numpy()
    b_single = label_array[b_logits.argmax(axis=1)]
    require(np.array_equal(b_single, saved.pred_model.to_numpy()), "B summary predictions differ")

    _, a_logits = read_scores(A_SNAPSHOT / "analysis/oof_predictions.csv", reference)
    a_single = label_array[a_logits.argmax(axis=1)]
    a_metrics = read_json(A_SNAPSHOT / "analysis/metrics.json")
    a_audit = read_json(ROOT / "docs/e30_embedding8b_fivefold_audit_aggregate.json")
    require(a_metrics["protocol_lock_sha256"] == EXPECTED["a_lock"]
            and a_audit["summary_oof_sha256"] == EXPECTED["a_oof"]
            and a_audit["gate_passed_against_e24"], "A independent audit is inconsistent")

    e24_metrics = read_json(ROOT / "outputs/e24_prvsiyan_analysis/metrics.json")
    source_paths = {
        "E24 E0c": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
        "RoBERTa": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
        "ModernBERT seed 42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
        "ModernBERT seed 17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
        "ModernBERT seed 73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
    }
    e24_expected = {
        "E24 E0c": e24_metrics["e24_scores_sha256"],
        "RoBERTa": e24_metrics["roberta_scores_sha256"],
        **{f"ModernBERT seed {seed}": e24_metrics["modern_seed_score_sha256"][str(seed)]
           for seed in (42, 17, 73)},
    }
    matrices = {}
    for name, path in source_paths.items():
        require(digest(path) == e24_expected[name] == a_metrics["e24_source_score_sha256"][name],
                f"E24 fixed source hash differs: {name}")
        _, matrices[name] = read_scores(path, reference)
    e24_frame = pd.read_csv(ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv",
                            dtype=str, keep_default_na=False)
    require(e24_frame[KEYS].equals(reference), "Archived E24 row identities differ")
    base = (row_z(matrices["E24 E0c"]) + row_z(matrices["RoBERTa"])
            + np.mean(np.stack([row_z(matrices[f"ModernBERT seed {seed}"])
                                for seed in (42, 17, 73)]), axis=0))
    e24_pred = label_array[base.argmax(axis=1)]
    require(np.array_equal(e24_pred, e24_frame.pred_e16_with_e24.to_numpy()),
            "Fixed E24 blend does not reproduce archived predictions")
    a_pred = label_array[(base + row_z(a_logits)).argmax(axis=1)]
    b_pred = label_array[(base + row_z(b_logits)).argmax(axis=1)]
    scores = {
        "e24_fixed": balanced(true, e24_pred),
        "e30_a_single": balanced(true, a_single),
        "e30_b_single": balanced(true, b_single),
        "e24_plus_e30_a_fixed": balanced(true, a_pred),
        "e24_plus_e30_b_fixed": balanced(true, b_pred),
    }
    close(scores["e24_fixed"], 0.425, "E24 BA")
    close(scores["e30_a_single"], a_audit["scores"]["e30_single"], "A single BA")
    close(scores["e24_plus_e30_a_fixed"], a_audit["scores"]["e24_plus_e30_fixed"], "E24+A BA")

    folds, classes = [], []
    for fold in range(1, 6):
        mask = reference.fold.to_numpy() == str(fold)
        rows = {"fold": fold}
        for name, pred in (("e30_a_single", a_single), ("e30_b_single", b_single),
                           ("e24_plus_e30_a_fixed", a_pred), ("e24_plus_e30_b_fixed", b_pred)):
            rows[name] = balanced(true[mask], pred[mask])
        rows["b_minus_a_fusion_delta"] = (rows["e24_plus_e30_b_fixed"]
                                          - rows["e24_plus_e30_a_fixed"])
        folds.append(rows)
    for label in LABELS:
        mask = true == label
        rows = {"model": label, "support": int(mask.sum())}
        for name, pred in (("e30_a_single_recall", a_single),
                           ("e30_b_single_recall", b_single),
                           ("e24_plus_e30_a_fixed_recall", a_pred),
                           ("e24_plus_e30_b_fixed_recall", b_pred)):
            rows[name] = float((pred[mask] == label).mean())
        rows["b_minus_a_fusion_delta"] = (rows["e24_plus_e30_b_fixed_recall"]
                                          - rows["e24_plus_e30_a_fixed_recall"])
        classes.append(rows)
    compare_csv(snapshot / "analysis/fold_metrics.csv", folds)
    compare_csv(snapshot / "analysis/class_metrics.csv", classes)

    confusion = np.zeros((18, 18), dtype=int)
    label_index = {label: index for index, label in enumerate(LABELS)}
    for actual, predicted in zip(true, b_single):
        confusion[label_index[actual], label_index[predicted]] += 1
    require(np.array_equal(confusion.sum(axis=1), np.full(18, 120)), "B confusion support invalid")
    gains = int(((a_pred != true) & (b_pred == true)).sum())
    losses = int(((a_pred == true) & (b_pred != true)).sum())

    # Each prompt has exactly one row per class; balanced accuracy is therefore
    # the mean correctness over all 2,160 development rows in every resample.
    prompt_delta = pd.DataFrame({
        "prompt": reference.prompt_id,
        "delta": (b_pred == true).astype(int) - (a_pred == true).astype(int),
    }).groupby("prompt", sort=True).delta.sum().to_numpy()
    require(len(prompt_delta) == 120, "Bootstrap prompt count changed")
    rng = np.random.default_rng(cfg["seed"])
    draws = np.empty(limits["paired_prompt_bootstrap_repeats"], dtype=np.float64)
    for start in range(0, len(draws), 500):
        stop = min(start + 500, len(draws))
        choices = rng.integers(0, 120, size=(stop - start, 120))
        draws[start:stop] = prompt_delta[choices].sum(axis=1) / 2160
    ci = np.quantile(draws, [0.025, 0.975]).tolist()

    saved_metrics_path = snapshot / "analysis/metrics.json"
    saved_metrics = read_json(saved_metrics_path)
    for key, wanted in {
        "experiment": cfg["experiment"], "stage": "fivefold_oof_analysis",
        "config_sha256": EXPECTED["b_config"], "script_sha256": EXPECTED["b_script"],
        "protocol_lock_sha256": digest(lock_path), "a_oof_sha256": EXPECTED["a_oof"],
        "e24_reference_oof_sha256": EXPECTED["e24_oof"], "rows": 2160,
        "prompts": 120, "labels": LABELS, "e30_b_single_confusion_matrix": confusion.tolist(),
        "b_fusion_wrong_to_right": gains, "b_fusion_right_to_wrong": losses,
        "holdout_evaluated": False, "fusion_weight_tuned": False,
        "deployment_and_kaggle_proven": False,
    }.items():
        require(saved_metrics.get(key) == wanted, f"B saved metrics differ in {key}")
    for key, value in scores.items():
        close(saved_metrics["balanced_accuracy"][key], value, f"B saved {key}")
    bootstrap = saved_metrics["paired_prompt_bootstrap_b_minus_a_fixed"]
    require(bootstrap["bootstrap_unit"] == "prompt_id" and bootstrap["repeats"] == 20_000
            and bootstrap["seed"] == 42
            and len(bootstrap["confidence_interval_95"]) == 2, "B bootstrap protocol changed")
    close(bootstrap["difference"], scores["e24_plus_e30_b_fixed"]
          - scores["e24_plus_e30_a_fixed"], "B bootstrap point difference")
    for actual, wanted in zip(bootstrap["confidence_interval_95"], ci):
        close(actual, wanted, "B bootstrap confidence interval")

    b_folds = [row["e24_plus_e30_b_fixed"] for row in folds]
    b_classes = [row["e24_plus_e30_b_fixed_recall"] for row in classes]
    a_folds = [row["e24_plus_e30_a_fixed"] for row in folds]
    a_classes = [row["e24_plus_e30_a_fixed_recall"] for row in classes]
    e24_folds = [balanced(true[reference.fold.to_numpy() == str(fold)],
                          e24_pred[reference.fold.to_numpy() == str(fold)]) for fold in range(1, 6)]
    e24_classes = [float((e24_pred[true == label] == label).mean()) for label in LABELS]
    main_comparison = comparison("E24+E30-A fixed", scores["e24_plus_e30_a_fixed"],
                                 a_folds, a_classes, scores["e24_plus_e30_b_fixed"],
                                 b_folds, b_classes, limits)
    require(saved_metrics["gate"] == main_comparison["gate"]
            and saved_metrics["gate_passed_vs_a"] == main_comparison["gate_passed"],
            "B saved structural promotion gate differs")
    close(saved_metrics["fusion_delta_vs_a"], main_comparison["balanced_accuracy_delta"],
          "B saved BA delta")
    close(saved_metrics["worst_class_recall_delta"], main_comparison["worst_class_recall_delta"],
          "B saved worst class drop")
    require(saved_metrics["folds_non_decreasing"] == main_comparison["folds_non_decreasing"],
            "B saved nondecreasing fold count differs")
    e24_comparison = comparison("E24 fixed", scores["e24_fixed"], e24_folds, e24_classes,
                                scores["e24_plus_e30_b_fixed"], b_folds, b_classes, limits)

    e28 = read_json(ROOT / "docs/e28_qwen8b_fivefold_audit_aggregate.json")
    close(e28["scores"]["e24_fixed"], scores["e24_fixed"], "E28 reference E24 BA")
    e28_folds = {int(row["fold"]): row["e24_plus_e28_fixed"] for row in e28["folds"]}
    e28_classes = {row["model"]: row["e24_plus_e28_fixed_recall"]
                   for row in e28["classes_recall"]}
    require(set(e28_folds) == set(range(1, 6)) and set(e28_classes) == set(LABELS),
            "E28 reference fold/class coverage differs")
    e28_comparison = comparison("E24+E28 fixed, unpromoted",
                                e28["scores"]["e24_plus_e28_fixed"],
                                [e28_folds[fold] for fold in range(1, 6)],
                                [e28_classes[label] for label in LABELS],
                                scores["e24_plus_e30_b_fixed"], b_folds, b_classes, limits)
    conservative = (main_comparison["gate_passed"] and e24_comparison["gate_passed"]
                    and e28_comparison["gate_passed"])

    aggregate = {
        "scope": "independent CPU audit of complete E30-B official development OOF; no holdout/test answers",
        "protocol_lock_sha256": digest(lock_path),
        "resource_report_sha256": digest(resource_path),
        "summary_oof_sha256": digest(summary_path),
        "summary_metrics_sha256": digest(saved_metrics_path),
        "reference_a_oof_sha256": EXPECTED["a_oof"],
        "reference_e24_oof_sha256": EXPECTED["e24_oof"],
        "locked_code_config_and_reported_model_data_hashes_match": True,
        "fivefold_oof_hashes_and_metadata_match": True,
        "fivefold_row_prompt_class_and_fold_integrity": True,
        "rows": 2160, "prompts": 120, "classes": 18,
        "resource_gate_passed": True,
        "resource_train_seconds": resource["train_seconds"],
        "resource_inference_4000_seconds": resource["inference_seconds"],
        "resource_train_peak_reserved_bytes": resource["train_peak_reserved_bytes"],
        "resource_inference_peak_reserved_bytes": resource["inference_peak_reserved_bytes"],
        "resource_update_fraction": (resource["completed_optimizer_updates"]
                                     / resource["planned_optimizer_updates"]),
        "fold_resources": fold_resources,
        "total_fold_train_seconds": sum(row["train_seconds"] for row in fold_resources),
        "max_fold_train_peak_reserved_bytes": max(row["train_peak_reserved_bytes"] for row in fold_resources),
        "max_fold_validation_peak_reserved_bytes": max(row["validation_peak_reserved_bytes"] for row in fold_resources),
        "scores": scores, "folds": folds, "class_recalls": classes,
        "e30_b_single_confusion_matrix": confusion.tolist(),
        "b_fusion_wrong_to_right": gains, "b_fusion_right_to_wrong": losses,
        "paired_prompt_bootstrap_b_minus_a_fixed": {
            "repeats": 20_000, "seed": cfg["seed"], "ci_95": ci,
        },
        "structural_comparison_vs_a": main_comparison,
        "other_locked_comparisons": {"e24_fixed": e24_comparison,
                                     "e28_unpromoted_fixed": e28_comparison},
        "gate_passed_vs_a": main_comparison["gate_passed"],
        "conservative_gate_passed_if_all_prior_fixed_fusions_count": conservative,
        "holdout_evaluated": False, "hidden_test_runtime_proven": False,
        "full_official_adapter_and_kaggle_package_proven": False,
    }
    output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"scores": scores, "structural_comparison_vs_a": main_comparison,
                      "conservative_gate_passed": conservative}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.snapshot, args.output)
