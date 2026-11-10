#!/usr/bin/env python3
"""E34 same-source E24 sparse-branch replacement; local CPU probe only.

Run prepare, baseline, then fit. The immutable lock and complete local E24
baseline report must exist before any four-view validation score is computed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score

from run_combined import make_pipeline as make_e24_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_e24_prvsiyan_external_ablation import select_external
from run_e33_four_view_sparse import make_four_view_pipeline, ordered_scores
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


ROOT = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
PREREG = ROOT / "docs/e34_four_view_e24_preregistration.md"
E24_SCRIPT = ROOT / "run_e24_prvsiyan_external_ablation.py"
E33_SCRIPT = ROOT / "run_e33_four_view_sparse.py"
E0C_SCRIPT = ROOT / "run_combined.py"
TRAIN = ROOT / "data/train.csv"
SOURCE = ROOT / "data/external/prvsiyan/passages.csv"
AUDIT_MANIFEST = ROOT / "data/external/prvsiyan/e24_accepted_manifest.csv"
AUDIT_REPORT = ROOT / "outputs/e24_prvsiyan_audit/report.json"
E0_DIR = ROOT / "outputs/e0_tfidf_svm"
E24_FIT = ROOT / "outputs/e24_prvsiyan_fit"
E24_ANALYSIS = ROOT / "outputs/e24_prvsiyan_analysis"
ROBERTA = ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"
MODERN = {seed: ROOT / path for seed, path in {
    42: "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
    17: "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
    73: "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
}.items()}
EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_E24_METRICS_SHA256 = "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557"
EXPECTED_E24_OOF_SHA256 = "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} differs from E24 on OOF key {key}")


def development():
    if sha256(TRAIN) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Official train.csv differs from E24")
    full = load_and_validate(TRAIN)
    dev, manifest, _, folds, _ = load_e0_split(full, E0_DIR, TRAIN)
    if (len(dev) != 2160 or dev["prompt_id"].nunique() != 120
            or len(manifest) != 150 or manifest["split"].eq("holdout").sum() != 30):
        raise ValueError("E0 development/holdout split changed")
    for fold in range(1, N_FOLDS + 1):
        fit = dev.iloc[np.flatnonzero(folds != fold)]
        valid = dev.iloc[np.flatnonzero(folds == fold)]
        if (len(fit) != 1728 or len(valid) != 432
                or fit["prompt_id"].nunique() != 96 or valid["prompt_id"].nunique() != 24
                or set(fit["prompt_id"]) & set(valid["prompt_id"])
                or set(fit["model"]) != set(LABELS) or set(valid["model"]) != set(LABELS)):
            raise ValueError(f"E0 fold {fold} changed or leaks prompts")
    return dev, folds


def external_rows():
    external, _, meta = select_external(SOURCE, AUDIT_MANIFEST, AUDIT_REPORT)
    saved = pd.read_csv(E24_FIT / "selected_external.csv", dtype=str, keep_default_na=False)
    columns = ("id", "prompt_id", "model", "served_model_id", "served_provider", "provider_route")
    if len(external) != len(saved):
        raise ValueError("E24 selected external row count changed")
    for column in columns:
        if not np.array_equal(external[column].to_numpy(), saved[column].to_numpy()):
            raise ValueError(f"E24 external selection changed in {column}")
    if not np.allclose(external["sample_weight"].to_numpy(dtype=float),
                       saved["sample_weight"].to_numpy(dtype=float), atol=1e-12, rtol=0):
        raise ValueError("E24 external sample weights changed")
    return external, meta


def reference(dev: pd.DataFrame):
    if sha256(E24_ANALYSIS / "metrics.json") != EXPECTED_E24_METRICS_SHA256:
        raise ValueError("Archived E24 analysis metrics changed")
    if sha256(E24_ANALYSIS / "oof_predictions.csv") != EXPECTED_E24_OOF_SHA256:
        raise ValueError("Archived E24 fixed-fusion OOF changed")
    metrics = read_json(E24_ANALYSIS / "metrics.json")
    paths = {
        "e24": (E24_FIT / "decision_scores.csv", metrics["e24_scores_sha256"]),
        "roberta": (ROBERTA, metrics["roberta_scores_sha256"]),
        **{f"modern_{seed}": (MODERN[seed], metrics["modern_seed_score_sha256"][str(seed)])
           for seed in (42, 17, 73)},
    }
    for name, (path, digest) in paths.items():
        if sha256(path) != digest:
            raise ValueError(f"Archived E24 source changed: {name}")
    e24 = load_scores(paths["e24"][0])
    if not np.array_equal(dev["id"].to_numpy(), e24["id"].to_numpy()):
        raise ValueError("Archived E24 scores differ from development row order")
    streams = {name: load_scores(path) for name, (path, _) in paths.items() if name != "e24"}
    saved = pd.read_csv(E24_ANALYSIS / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for name, frame in [*streams.items(), ("saved E24 OOF", saved)]:
        aligned(e24, frame, name)
    other = row_standardize(streams["roberta"].loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
    other += np.mean(np.stack([
        row_standardize(streams[f"modern_{seed}"].loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
        for seed in (42, 17, 73)
    ]), axis=0)
    e24_scores = e24.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    labels = np.asarray(LABELS)
    archived_sparse = labels[e24_scores.argmax(axis=1)]
    archived_fusion = labels[(row_standardize(e24_scores) + other).argmax(axis=1)]
    if not np.array_equal(archived_sparse, saved["pred_e24_e0c"].to_numpy()):
        raise ValueError("Archived E24 sparse predictions do not reproduce")
    if not np.array_equal(archived_fusion, saved["pred_e16_with_e24"].to_numpy()):
        raise ValueError("Archived E24 fusion predictions do not reproduce")
    if not np.isclose(balanced_accuracy_score(e24["true_model"], archived_fusion), 0.425, atol=1e-12):
        raise ValueError("Archived E24 fixed-fusion BA does not reproduce")
    return e24, other, archived_sparse, archived_fusion


def lock_contents() -> dict:
    return {
        "experiment": "e34_four_view_e24_same_source",
        "protocol_version": 1,
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(SCRIPT),
        "e24_script_sha256": sha256(E24_SCRIPT),
        "e33_script_sha256": sha256(E33_SCRIPT),
        "e0c_script_sha256": sha256(E0C_SCRIPT),
        "official_train_sha256": sha256(TRAIN),
        "raw_external_sha256": sha256(SOURCE),
        "audit_manifest_sha256": sha256(AUDIT_MANIFEST),
        "audit_report_sha256": sha256(AUDIT_REPORT),
        "selected_external_sha256": sha256(E24_FIT / "selected_external.csv"),
        "e0_manifest_sha256": sha256(E0_DIR / "split_manifest.csv"),
        "e0_oof_sha256": sha256(E0_DIR / "oof_predictions.csv"),
        "e24_metrics_sha256": sha256(E24_ANALYSIS / "metrics.json"),
        "e24_oof_sha256": sha256(E24_ANALYSIS / "oof_predictions.csv"),
        "e24_sparse_scores_sha256": sha256(E24_FIT / "decision_scores.csv"),
        "roberta_scores_sha256": sha256(ROBERTA),
        "modern_scores_sha256": {str(seed): sha256(MODERN[seed]) for seed in (42, 17, 73)},
    }


def checked_lock(output_dir: Path) -> dict:
    lock = read_json(output_dir / "protocol_lock.json")
    if lock != lock_contents():
        raise ValueError("E34 code, preregistration, or input changed after protocol lock")
    return lock


def prepare(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("E34 output directory already contains files")
    dev, _ = development()
    external, meta = external_rows()
    reference(dev)
    lock = lock_contents()
    write_new_json(output_dir / "protocol_lock.json", lock)
    print(json.dumps({"stage": "prepare", "protocol_lock_sha256": sha256(output_dir / "protocol_lock.json"),
                      "selected_external_rows": len(external), "e34_scored": False}, indent=2), flush=True)


def baseline(output_dir: Path) -> None:
    checked_lock(output_dir)
    baseline_dir = output_dir / "baseline"
    if baseline_dir.exists():
        raise ValueError("E34 local E24 baseline already exists")
    dev, folds = development()
    external, _ = external_rows()
    archived, other, archived_sparse, archived_fusion = reference(dev)
    texts, truth = dev["text"].to_numpy(), dev["model"].to_numpy()
    ext_text, ext_label = external["text"].to_numpy(), external["model"].to_numpy()
    ext_weight = external["sample_weight"].to_numpy(dtype=np.float64)
    local_scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    for fold in range(1, N_FOLDS + 1):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = make_e24_pipeline()
        model.fit(np.concatenate((texts[fit_idx], ext_text)),
                  np.concatenate((truth[fit_idx], ext_label)),
                  classifier__sample_weight=np.concatenate((np.ones(len(fit_idx)), ext_weight)))
        local_scores[valid_idx] = ordered_scores(model, texts[valid_idx])
        print(f"E34 local E24 baseline fold {fold} fitted", flush=True)
        del model
    if not np.isfinite(local_scores).all():
        raise ValueError("Local E24 baseline scores are incomplete")
    labels = np.asarray(LABELS)
    local_sparse = labels[local_scores.argmax(axis=1)]
    local_fusion = labels[(row_standardize(local_scores.astype(np.float64)) + other).argmax(axis=1)]
    archived_scores = archived.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    mismatch = int(np.sum(local_sparse != archived_sparse))
    largest_difference = float(np.max(np.abs(local_scores.astype(np.float64) - archived_scores)))
    reproduced = mismatch == 0 and largest_difference <= 1e-5
    baseline_dir.mkdir(parents=True, exist_ok=False)
    score_frame = archived.loc[:, list(KEY_COLUMNS)].copy()
    for index, label in enumerate(LABELS):
        score_frame[f"score_{label}"] = local_scores[:, index]
    score_frame.to_csv(baseline_dir / "decision_scores.csv", index=False)
    pred_frame = archived.loc[:, list(KEY_COLUMNS)].copy()
    pred_frame["pred_local_e24_sparse"] = local_sparse
    pred_frame["pred_local_e24_fusion"] = local_fusion
    pred_frame.to_csv(baseline_dir / "oof_predictions.csv", index=False)
    write_new_json(baseline_dir / "metrics.json", {
        "experiment": "e34_four_view_e24_same_source", "stage": "local_e24_baseline_before_e34",
        "protocol_lock_sha256": sha256(output_dir / "protocol_lock.json"),
        "python_version": sys.version.split()[0], "sklearn_version": sklearn.__version__,
        "selected_external_rows": len(external), "development_rows": len(dev),
        "prediction_mismatch_vs_archived_e24": mismatch,
        "max_abs_margin_difference_vs_archived_e24": largest_difference,
        "archived_e24_reproduced_locally": reproduced,
        "archived_e24_sparse_ba": float(balanced_accuracy_score(truth, archived_sparse)),
        "local_e24_sparse_ba": float(balanced_accuracy_score(truth, local_sparse)),
        "archived_e24_fusion_ba": float(balanced_accuracy_score(truth, archived_fusion)),
        "local_e24_fusion_ba": float(balanced_accuracy_score(truth, local_fusion)),
        "decision_scores_sha256": sha256(baseline_dir / "decision_scores.csv"),
        "oof_predictions_sha256": sha256(baseline_dir / "oof_predictions.csv"),
        "holdout_scored": False, "test_scored": False,
    })
    print(json.dumps({"stage": "baseline", "mismatch_rows": mismatch,
                      "max_abs_margin_difference": largest_difference,
                      "reproduced": reproduced}, indent=2), flush=True)


def fit(output_dir: Path) -> None:
    checked_lock(output_dir)
    candidate_dir = output_dir / "candidate"
    if candidate_dir.exists():
        raise ValueError("E34 candidate output already exists")
    baseline_dir = output_dir / "baseline"
    baseline_meta = read_json(baseline_dir / "metrics.json")
    if (baseline_meta.get("stage") != "local_e24_baseline_before_e34"
            or baseline_meta.get("protocol_lock_sha256") != sha256(output_dir / "protocol_lock.json")
            or baseline_meta.get("decision_scores_sha256") != sha256(baseline_dir / "decision_scores.csv")
            or baseline_meta.get("oof_predictions_sha256") != sha256(baseline_dir / "oof_predictions.csv")):
        raise ValueError("Complete local E24 baseline report is required before E34 fit")
    dev, folds = development()
    external, _ = external_rows()
    archived, other, archived_sparse, archived_fusion = reference(dev)
    local = load_scores(baseline_dir / "decision_scores.csv")
    aligned(archived, local, "Local E24 baseline")
    local_scores = local.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    labels = np.asarray(LABELS)
    local_sparse = labels[local_scores.argmax(axis=1)]
    local_fusion = labels[(row_standardize(local_scores) + other).argmax(axis=1)]
    saved_baseline_predictions = pd.read_csv(baseline_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    if (not np.array_equal(local_sparse, saved_baseline_predictions["pred_local_e24_sparse"].to_numpy())
            or not np.array_equal(local_fusion, saved_baseline_predictions["pred_local_e24_fusion"].to_numpy())):
        raise ValueError("Local E24 baseline margins and predictions disagree")
    texts, truth = dev["text"].to_numpy(), dev["model"].to_numpy()
    ext_text, ext_label = external["text"].to_numpy(), external["model"].to_numpy()
    ext_weight = external["sample_weight"].to_numpy(dtype=np.float64)
    four_scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    for fold in range(1, N_FOLDS + 1):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = make_four_view_pipeline()
        model.fit(np.concatenate((texts[fit_idx], ext_text)),
                  np.concatenate((truth[fit_idx], ext_label)),
                  classifier__sample_weight=np.concatenate((np.ones(len(fit_idx)), ext_weight)))
        four_scores[valid_idx] = ordered_scores(model, texts[valid_idx])
        print(f"E34 four-view fold {fold} fitted", flush=True)
        del model
    if not np.isfinite(four_scores).all():
        raise ValueError("E34 OOF scores are incomplete")
    four_sparse = labels[four_scores.argmax(axis=1)]
    four_fusion = labels[(row_standardize(four_scores.astype(np.float64)) + other).argmax(axis=1)]
    candidates = {
        "archived_e24_sparse": archived_sparse,
        "local_e24_sparse": local_sparse,
        "e34_four_view_sparse": four_sparse,
        "archived_e24_fusion": archived_fusion,
        "local_e24_fusion": local_fusion,
        "e34_four_view_fusion": four_fusion,
    }
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        mask = folds == fold
        row = {"fold": fold, "rows": int(mask.sum()),
               **{name: float(balanced_accuracy_score(truth[mask], pred[mask]))
                  for name, pred in candidates.items()}}
        row["same_source_fusion_delta"] = row["e34_four_view_fusion"] - row["local_e24_fusion"]
        fold_rows.append(row)
    class_rows = []
    for label in LABELS:
        mask = truth == label
        row = {"model": label, "support": int(mask.sum()),
               **{name: float((pred[mask] == label).mean()) for name, pred in candidates.items()}}
        row["same_source_fusion_delta"] = row["e34_four_view_fusion"] - row["local_e24_fusion"]
        class_rows.append(row)
    fold_frame, class_frame = pd.DataFrame(fold_rows), pd.DataFrame(class_rows)
    pooled = {name: float(balanced_accuracy_score(truth, pred)) for name, pred in candidates.items()}
    delta = pooled["e34_four_view_fusion"] - pooled["local_e24_fusion"]
    nondecrease = int((fold_frame["same_source_fusion_delta"] >= -1e-12).sum())
    worst = float(class_frame["same_source_fusion_delta"].min())
    gate = {
        "local_e24_reproduced_archived": bool(baseline_meta["archived_e24_reproduced_locally"]),
        "fusion_delta_at_least_0_01": delta >= 0.01 - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": nondecrease >= 4,
        "no_class_recall_drop_at_least_0_05": worst > -0.05 + 1e-12,
    }
    metrics = {
        "experiment": "e34_four_view_e24_same_source", "stage": "five_fold_oof_analysis",
        "status": "confirmatory_local" if baseline_meta["archived_e24_reproduced_locally"] else "exploratory_environment_mismatch",
        "protocol_lock_sha256": sha256(output_dir / "protocol_lock.json"),
        "baseline_metrics_sha256": sha256(baseline_dir / "metrics.json"),
        "python_version": sys.version.split()[0], "sklearn_version": sklearn.__version__,
        "development_rows": len(dev), "development_prompts": int(dev["prompt_id"].nunique()),
        "external_rows_per_fit": len(external), "pooled_balanced_accuracy": pooled,
        "same_source_fusion_delta": delta, "fusion_folds_non_decreasing": nondecrease,
        "worst_class_recall_delta": worst,
        "worst_class": str(class_frame.loc[class_frame["same_source_fusion_delta"].idxmin(), "model"]),
        "paired_prompt_bootstrap_vs_local_e24": paired_prompt_bootstrap(
            truth, dev["prompt_id"].to_numpy(), local_fusion, four_fusion,
            repeats=20_000, seed=42),
        "gate": gate, "gate_passed": bool(all(gate.values())),
        "source_protocol_exploratory_due_to_global_official_length_statistics": True,
        "weights_tuned": False, "holdout_scored": False, "test_scored": False,
    }
    candidate_dir.mkdir(parents=True, exist_ok=False)
    score_frame = archived.loc[:, list(KEY_COLUMNS)].copy()
    for index, label in enumerate(LABELS):
        score_frame[f"score_{label}"] = four_scores[:, index]
    score_frame.to_csv(candidate_dir / "decision_scores.csv", index=False)
    pred_frame = archived.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in candidates.items():
        pred_frame[f"pred_{name}"] = pred
    pred_frame.to_csv(candidate_dir / "oof_predictions.csv", index=False)
    fold_frame.to_csv(candidate_dir / "fold_metrics.csv", index=False)
    class_frame.to_csv(candidate_dir / "class_metrics.csv", index=False)
    write_new_json(candidate_dir / "metrics.json", metrics)
    print(json.dumps({"stage": "fit", "status": metrics["status"],
                      "pooled_balanced_accuracy": pooled, "gate": gate}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "baseline", "fit"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e34_four_view_e24")
    args = parser.parse_args()
    {"prepare": prepare, "baseline": baseline, "fit": fit}[args.stage](args.output_dir)


if __name__ == "__main__":
    main()
