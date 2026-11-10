#!/usr/bin/env python3
"""E36 preregistered CPU-only E24/E28/E30 fixed OOF fusion evaluation.

Run ``lock`` before any E30 labeled OOF exists. ``evaluate`` requires that
immutable lock and writes aggregate diagnostics only. No model or GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "docs" / "e36_fixed_triple_fusion_protocol_lock_v2.json"
DOC = ROOT / "docs" / "e36_fixed_triple_fusion_preregistration.md"
OUT = ROOT / "outputs" / "e36_fixed_triple_fusion"
E28_DIR = ROOT / "outputs" / "e28_qwen8b_fivefold"
E30_DIR = ROOT / "outputs" / "e30_qwen3_embedding8b_v3"
E24_METRICS_SHA = "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557"
E24_OOF_SHA = "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448"
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
E28_OOF_SHA = "9243bb2002158c3f3078fcbba6f8746db71c75ee2ac2807a6789104b4d310d64"
E28_METRICS_SHA = "ae4e073357f7b33946d526d87c42c33fa74e33f551c63c5a97c1dfe21479f98f"
FOLDS = (1, 2, 3, 4, 5)
BRANCHES = ("e24_fixed", "e24_plus_e28", "e24_plus_e30", "triple")
COMPARATORS = BRANCHES[:3]
SOURCES = {
    "e24_oof": ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv",
    "e24_scores": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
    "roberta_scores": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
    "modern_seed_42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
    "modern_seed_17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
    "modern_seed_73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_hash(path: Path) -> str:
    value = json.dumps(read_json(path), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalized_text_hash(path: Path) -> str:
    value = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def expected_source_hashes() -> dict[str, str]:
    metrics_path = ROOT / "outputs/e24_prvsiyan_analysis/metrics.json"
    if sha256(metrics_path) != E24_METRICS_SHA:
        raise ValueError("Archived E24 metrics SHA-256 changed")
    metrics = read_json(metrics_path)
    return {
        "e24_oof": E24_OOF_SHA,
        "e24_scores": metrics["e24_scores_sha256"],
        "roberta_scores": metrics["roberta_scores_sha256"],
        **{f"modern_seed_{seed}": metrics["modern_seed_score_sha256"][str(seed)]
           for seed in (42, 17, 73)},
    }


def source_fingerprint() -> dict:
    expected = expected_source_hashes()
    for name, path in SOURCES.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Archived E24 source SHA-256 changed: {name}")
    if sha256(ROOT / "data/train.csv") != TRAIN_SHA:
        raise ValueError("Official train.csv SHA-256 changed")
    e28 = read_json(E28_DIR / "protocol_lock.json")
    e30 = read_json(E30_DIR / "protocol_lock.json")
    for name, protocol, version in (("E28", e28, 1), ("E30", e30, 3)):
        if protocol.get("protocol_version") != version or protocol.get("train_sha256") != TRAIN_SHA:
            raise ValueError(f"{name} protocol version or training data changed")
        if protocol.get("e24_metrics_sha256", E24_METRICS_SHA) != E24_METRICS_SHA:
            raise ValueError(f"{name} E24 reference changed")
    if e28.get("e24_sources_sha256") != expected:
        raise ValueError("E28 lock E24 source hashes changed")
    if e30.get("model_repo") != "Qwen/Qwen3-Embedding-8B":
        raise ValueError("E30 lock model identity changed")
    files = {
        "e28_script": ROOT / "run_e28_qwen8b_fivefold.py",
        "e30_script": ROOT / "run_e30_supervised_embedding8b.py",
        "e36_script": Path(__file__).resolve(),
        "e36_preregistration": DOC,
    }
    return {
        "official_train_sha256": TRAIN_SHA,
        "e0_manifest_sha256": sha256(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv"),
        "e0_oof_sha256": sha256(ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv"),
        "e24_metrics_sha256": E24_METRICS_SHA,
        "completed_e28_analysis_oof_sha256": E28_OOF_SHA,
        "completed_e28_analysis_metrics_sha256": E28_METRICS_SHA,
        "e24_source_sha256": expected,
        "e28_protocol_canonical_sha256": canonical_json_hash(E28_DIR / "protocol_lock.json"),
        "e30_protocol_canonical_sha256": canonical_json_hash(E30_DIR / "protocol_lock.json"),
        "e28_config_canonical_sha256": canonical_json_hash(ROOT / "configs/e28_qwen3_8b_fivefold.json"),
        "e30_config_canonical_sha256": canonical_json_hash(ROOT / "configs/e30_qwen3_embedding8b.json"),
        "normalized_text_sha256": {name: normalized_text_hash(path) for name, path in files.items()},
    }


def no_e30_oof() -> None:
    paths = [E30_DIR / "analysis/oof_predictions.csv",
             *[E30_DIR / f"fold_{fold}/oof_predictions.csv" for fold in FOLDS]]
    if any(path.exists() for path in paths):
        raise ValueError("E30 labeled OOF already exists; prospective lock is too late")


def lock() -> None:
    if LOCK.exists():
        raise ValueError("E36 protocol lock already exists; refusing to overwrite")
    no_e30_oof()
    fingerprint = source_fingerprint()  # hashes only; no E30 OOF read
    value = {
        "experiment": "e36_fixed_e24_e28_e30_triple_fusion",
        "protocol_version": 2,
        "stage": "prospective_lock_before_e30_labeled_oof",
        "supersedes_v1_lock_sha256": sha256(ROOT / "docs/e36_fixed_triple_fusion_protocol_lock.json"),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_fingerprint": fingerprint,
        "fusion": "E24 three-branch archived row-z score sum + 1.0*row-z(E28 logits) + 1.0*row-z(E30 logits)",
        "comparisons": list(BRANCHES),
        "selection": "strongest comparator = highest pooled OOF BA among E24, E24+E28, E24+E30; ties in that order",
        "gate": {"min_balanced_accuracy_delta": 0.01,
                 "min_non_decreasing_folds": 4,
                 "max_allowed_class_recall_drop": 0.05},
        "paired_prompt_bootstrap": {"repeats": 20000, "seed": 42},
        "e30_oof_read": False,
        "holdout_or_test_scored": False,
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"lock": str(LOCK), "sha256": sha256(LOCK),
                      "locked_at_utc": value["locked_at_utc"]}, indent=2))


def aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"{name} row count changed")
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} differs in {key}")


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)


def verify_completed_branch(folder: Path, name: str, reference: pd.DataFrame) -> pd.DataFrame:
    protocol_path = folder / "protocol_lock.json"
    metrics = read_json(folder / "analysis/metrics.json")
    protocol_sha = sha256(protocol_path)
    protocol = read_json(protocol_path)
    cfg = read_json(ROOT / ("configs/e28_qwen3_8b_fivefold.json" if name == "E28"
                            else "configs/e30_qwen3_embedding8b.json"))
    if protocol.get("experiment") != cfg["experiment"] or protocol.get("model_revision") != cfg["model_revision"]:
        raise ValueError(f"{name} protocol model identity changed")
    if name == "E28":
        if sha256(folder / "analysis/oof_predictions.csv") != E28_OOF_SHA:
            raise ValueError("Completed E28 analysis OOF changed after prospective lock")
        if sha256(folder / "analysis/metrics.json") != E28_METRICS_SHA:
            raise ValueError("Completed E28 analysis metrics changed after prospective lock")
    if metrics.get("protocol_lock_sha256") != protocol_sha:
        raise ValueError(f"{name} analysis/protocol SHA-256 mismatch")
    if metrics.get("e24_reference_oof_sha256") != E24_OOF_SHA:
        raise ValueError(f"{name} E24 reference SHA-256 mismatch")
    if metrics.get("rows") != 2160 or metrics.get("prompts") != 120 or metrics.get("labels") != list(LABELS):
        raise ValueError(f"{name} analysis dimensions or labels changed")
    pieces = []
    resource = (folder / "resource_report.json") if name == "E30" else ROOT / "outputs/e28_qwen8b_resource_mirror/resource_report.json"
    resource_report = read_json(resource)
    expected_resource_experiment = ("e28_qwen3_8b_lora_feasibility" if name == "E28"
                                    else cfg["experiment"])
    if (resource_report.get("experiment") != expected_resource_experiment
            or not resource_report.get("gate_passed")):
        raise ValueError(f"{name} resource gate report changed or failed")
    for fold in FOLDS:
        fold_path = folder / f"fold_{fold}/oof_predictions.csv"
        run = read_json(folder / f"fold_{fold}/run.json")
        if (run.get("experiment") != cfg["experiment"] or run.get("fold") != fold
                or run.get("protocol_lock_sha256") != protocol_sha
                or run.get("model_revision") != cfg["model_revision"]
                or run.get("train_rows") != 1728 or run.get("validation_rows") != 432
                or run.get("no_epoch_or_threshold_selection") is not True
                or run.get("holdout_evaluated") is not False):
            raise ValueError(f"{name} fold {fold} protocol metadata mismatch")
        if name == "E28":
            if run.get("seed") != 42 or run.get("training_epochs_completed") != cfg["epochs"]:
                raise ValueError(f"E28 fold {fold} seed or completed epochs changed")
        elif (run.get("config_sha256") != sha256(ROOT / "configs/e30_qwen3_embedding8b.json")
              or run.get("epochs") != cfg["epochs"]
              or len(run.get("epoch_loss", [])) != cfg["epochs"]
              or run.get("completed_optimizer_updates", 0) <
              0.95 * run.get("planned_optimizer_updates", 0)):
            raise ValueError(f"E30 fold {fold} config, epochs, or updates changed")
        if run.get("oof_predictions_sha256") != sha256(fold_path):
            raise ValueError(f"{name} fold {fold} OOF SHA-256 mismatch")
        if run.get("resource_report_sha256") != sha256(resource):
            raise ValueError(f"{name} fold {fold} resource report SHA-256 mismatch")
        piece = load_scores(fold_path)
        if len(piece) != 432 or set(piece["fold"]) != {str(fold)}:
            raise ValueError(f"{name} fold {fold} dimensions changed")
        if name == "E28" and set(piece["seed"]) != {42}:
            raise ValueError(f"E28 fold {fold} seed changed")
        if "pred_model" not in piece or not np.array_equal(
                piece["pred_model"].to_numpy(), np.asarray(LABELS)[matrix(piece).argmax(axis=1)]):
            raise ValueError(f"{name} fold {fold} saved predictions differ from logits")
        pieces.append(piece)
    merged = pd.concat(pieces, ignore_index=True).set_index("id").loc[reference["id"]].reset_index()
    analysis = load_scores(folder / "analysis/oof_predictions.csv")
    aligned(reference, merged, f"{name} fold OOF")
    aligned(reference, analysis, f"{name} analysis OOF")
    if not np.array_equal(matrix(merged), matrix(analysis)):
        raise ValueError(f"{name} fold and analysis scores differ")
    return analysis


def evaluate() -> None:
    locked = read_json(LOCK)
    if locked.get("experiment") != "e36_fixed_e24_e28_e30_triple_fusion" or locked.get("protocol_version") != 2:
        raise ValueError("E36 protocol lock identity changed")
    if locked.get("source_fingerprint") != source_fingerprint():
        raise ValueError("E36 code, preregistration, source, or protocol differs from immutable lock")
    if OUT.exists():
        raise ValueError("E36 output already exists; refusing overwrite")
    reference = pd.read_csv(SOURCES["e24_oof"], dtype=str, keep_default_na=False)
    if len(reference) != 2160 or reference["prompt_id"].nunique() != 120:
        raise ValueError("Archived E24 development dimensions changed")
    if set(reference["fold"]) != {str(fold) for fold in FOLDS}:
        raise ValueError("Archived E24 fold assignments changed")
    e28 = verify_completed_branch(E28_DIR, "E28", reference)
    e30 = verify_completed_branch(E30_DIR, "E30", reference)
    inputs = {name: load_scores(path) for name, path in SOURCES.items() if name != "e24_oof"}
    for name, frame in inputs.items():
        aligned(reference, frame, name)
    e24_scores = (row_standardize(matrix(inputs["e24_scores"]))
                  + row_standardize(matrix(inputs["roberta_scores"]))
                  + np.mean(np.stack([row_standardize(matrix(inputs[f"modern_seed_{seed}"]))
                                      for seed in (42, 17, 73)]), axis=0))
    labels = np.asarray(LABELS)
    e28_z, e30_z = row_standardize(matrix(e28)), row_standardize(matrix(e30))
    score_map = {"e24_fixed": e24_scores,
                 "e24_plus_e28": e24_scores + e28_z,
                 "e24_plus_e30": e24_scores + e30_z,
                 "triple": e24_scores + e28_z + e30_z}
    pred = {name: labels[scores.argmax(axis=1)] for name, scores in score_map.items()}
    if not np.array_equal(pred["e24_fixed"], reference["pred_e16_with_e24"].to_numpy()):
        raise ValueError("Archived E24 row-level predictions failed exact reproduction")
    true = reference["true_model"].to_numpy()
    if not np.isclose(balanced_accuracy_score(true, pred["e24_fixed"]), 0.425, atol=1e-12):
        raise ValueError("Archived E24 BA failed exact reproduction")
    ba = {name: float(balanced_accuracy_score(true, pred[name])) for name in BRANCHES}
    e28_metrics = read_json(E28_DIR / "analysis/metrics.json")
    e30_metrics = read_json(E30_DIR / "analysis/metrics.json")
    if not np.isclose(ba["e24_plus_e28"], e28_metrics["balanced_accuracy"]["e24_plus_e28_fixed"], atol=1e-12):
        raise ValueError("E28 fixed comparator did not reproduce its saved metric")
    if not np.isclose(ba["e24_plus_e30"], e30_metrics["balanced_accuracy"]["e24_plus_e30_fixed"], atol=1e-12):
        raise ValueError("E30 fixed comparator did not reproduce its saved metric")
    fold = reference["fold"].to_numpy()
    fold_rows = [{"fold": f, **{name: float(balanced_accuracy_score(true[fold == str(f)],
                           pred[name][fold == str(f)])) for name in BRANCHES}} for f in FOLDS]
    class_rows = [{"model": label, "support": int((true == label).sum()),
                   **{f"{name}_recall": float((pred[name][true == label] == label).mean())
                      for name in BRANCHES}} for label in LABELS]
    strongest = max(COMPARATORS, key=lambda name: ba[name])
    delta = ba["triple"] - ba[strongest]
    nondecrease = sum(row["triple"] >= row[strongest] - 1e-12 for row in fold_rows)
    worst_class_delta = min(row["triple_recall"] - row[f"{strongest}_recall"] for row in class_rows)
    gate = {"delta_at_least_0_01": bool(delta >= 0.01 - 1e-12),
            "at_least_four_folds_non_decreasing": bool(nondecrease >= 4),
            "no_class_recall_drop_at_least_0_05": bool(worst_class_delta > -0.05 + 1e-12)}
    comparisons = {}
    prompts = reference["prompt_id"].to_numpy()
    for name in COMPARATORS:
        old, new = pred[name], pred["triple"]
        comparisons[name] = {
            "balanced_accuracy_delta": ba["triple"] - ba[name],
            "prediction_changed": int((old != new).sum()),
            "wrong_to_right": int(((old != true) & (new == true)).sum()),
            "right_to_wrong": int(((old == true) & (new != true)).sum()),
            "wrong_to_different_wrong": int(((old != true) & (new != true) & (old != new)).sum()),
            "paired_prompt_bootstrap": paired_prompt_bootstrap(true, prompts, old, new,
                                                               repeats=20000, seed=42),
        }
    report = {
        "experiment": locked["experiment"], "stage": "five_fold_oof_analysis",
        "protocol_lock_sha256": sha256(LOCK),
        "input_sha256": {"e28_oof": sha256(E28_DIR / "analysis/oof_predictions.csv"),
                         "e28_metrics": sha256(E28_DIR / "analysis/metrics.json"),
                         "e30_oof": sha256(E30_DIR / "analysis/oof_predictions.csv"),
                         "e30_metrics": sha256(E30_DIR / "analysis/metrics.json")},
        "rows": len(reference), "prompts": int(reference["prompt_id"].nunique()),
        "labels": list(LABELS), "balanced_accuracy": ba,
        "fold_metrics": fold_rows, "class_metrics": class_rows,
        "strongest_fixed_comparator": strongest,
        "triple_minus_strongest": {"balanced_accuracy_delta": delta,
                                     "non_decreasing_folds": nondecrease,
                                     "worst_class_recall_delta": worst_class_delta,
                                     "gate": gate, "gate_passed": all(gate.values())},
        "triple_vs_each_comparator": comparisons,
        "holdout_evaluated": False, "test_evaluated": False,
        "fusion_weight_or_threshold_tuned": False,
        "row_level_predictions_exported": False,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "class_metrics.csv", index=False)
    print(json.dumps({"balanced_accuracy": ba, "strongest_fixed_comparator": strongest,
                      "gate": gate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "evaluate"))
    args = parser.parse_args()
    {"lock": lock, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
