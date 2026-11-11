#!/usr/bin/env python3
"""E40: train-fold prompt-centered style features with raw scaled inference.

`preflight` checks the locked official development partition without fitting or
opening E24 class scores. `run` reproduces E24, then runs one fixed five-fold
ordinary-versus-residual style comparison. Neither mode scores holdout/test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_e0 import LABELS
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import FEATURE_NAMES, extract_features


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data/train.csv"
E0_DIR = ROOT / "outputs/e0_tfidf_svm"
PREREG = ROOT / "docs/e40_prompt_residual_style_preregistration.md"
OUT = ROOT / "outputs/e40_prompt_residual_style"
EXPECTED_TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_MANIFEST_SHA = "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15"
EXPECTED_E0_OOF_SHA = "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa"
FOLDS = (1, 2, 3, 4, 5)
FUSION_WEIGHT = 0.5
SEED = 42
BOOTSTRAP_REPEATS = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_development() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Stream official rows; keep only the locked 120 development prompts."""
    if not PREREG.is_file() or PREREG.stat().st_size == 0:
        raise ValueError("E40 preregistration must exist before preflight")
    if len(FEATURE_NAMES) != 57 or len(set(FEATURE_NAMES)) != 57:
        raise ValueError("The original E0b 57-feature definition changed")
    manifest_path = E0_DIR / "split_manifest.csv"
    e0_oof_path = E0_DIR / "oof_predictions.csv"
    if sha256(TRAIN) != EXPECTED_TRAIN_SHA:
        raise ValueError("Official train.csv hash changed")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise ValueError("E0 split manifest hash changed")
    if sha256(e0_oof_path) != EXPECTED_E0_OOF_SHA:
        raise ValueError("E0 OOF identity source hash changed")

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"] or manifest["prompt_id"].duplicated().any():
        raise ValueError("E0 split manifest structure changed")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("E0 split labels changed")
    dev_prompts = set(manifest.loc[manifest["split"].eq("development"), "prompt_id"])
    if len(manifest) != 150 or len(dev_prompts) != 120 or manifest["split"].eq("holdout").sum() != 30:
        raise ValueError("Unexpected E0 development/holdout prompt counts")

    rows: list[dict[str, str]] = []
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"id", "prompt_id", "text", "model"}:
            raise ValueError("Official training columns changed")
        for row in reader:
            if row["prompt_id"] in dev_prompts:
                rows.append({key: row[key] for key in ("id", "prompt_id", "text", "model")})
    dev = pd.DataFrame(rows)
    e0 = pd.read_csv(e0_oof_path, dtype=str, keep_default_na=False,
                     usecols=["id", "prompt_id", "true_model", "fold"])
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if len(dev) != 2160 or not dev["id"].is_unique or dev["prompt_id"].nunique() != 120:
        raise ValueError("E40 development dimensions changed")
    if not expected.reset_index(drop=True).equals(e0.loc[:, ["id", "prompt_id", "true_model"]]):
        raise ValueError("Official development rows differ from E0 OOF identity")
    if set(dev["model"]) != set(LABELS) or not dev.groupby("model").size().eq(120).all():
        raise ValueError("E40 development class counts changed")
    if not dev.groupby("prompt_id").size().eq(18).all():
        raise ValueError("E40 development prompt counts changed")
    folds = pd.to_numeric(e0["fold"], errors="raise").to_numpy(dtype=np.int64)
    if set(folds) != set(FOLDS):
        raise ValueError("E0 fold IDs changed")
    for fold in FOLDS:
        fit = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        if len(fit) != 1728 or len(valid) != 432:
            raise ValueError(f"Fold {fold} has unexpected row counts")
        if set(dev.iloc[fit]["prompt_id"]) & set(dev.iloc[valid]["prompt_id"]):
            raise ValueError(f"Fold {fold} overlaps prompt groups")
        if dev.iloc[valid]["prompt_id"].nunique() != 24:
            raise ValueError(f"Fold {fold} has unexpected prompt groups")
        if not dev.iloc[valid].groupby("model").size().eq(24).all():
            raise ValueError(f"Fold {fold} has unexpected class support")

    checks = {
        "official_train_sha256": EXPECTED_TRAIN_SHA,
        "e0_split_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "e0_oof_sha256": EXPECTED_E0_OOF_SHA,
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__).resolve()),
        "style_source_sha256": sha256(ROOT / "run_style.py"),
        "e24_reproducer_sha256": sha256(ROOT / "run_e37_char4_likelihood.py"),
        "n_development_rows": len(dev),
        "n_development_prompts": int(dev["prompt_id"].nunique()),
        "n_features": len(FEATURE_NAMES),
        "folds": len(FOLDS),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "holdout_or_test_rows_scored": False,
    }
    return dev, folds, checks


def classifier() -> LinearSVC:
    return LinearSVC(C=1.0, class_weight="balanced", dual=False,
                     max_iter=10_000, random_state=SEED)


def ordered_scores(model: LinearSVC, values: np.ndarray) -> np.ndarray:
    positions = [int(np.flatnonzero(model.classes_ == label)[0]) for label in LABELS]
    scores = model.decision_function(values)[:, positions].astype(np.float64)
    if scores.shape != (len(values), len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError("E40 style decision scores are non-finite or misaligned")
    return scores


def residualize_train(values: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """Subtract prompt means from scaled outer-train rows only."""
    groups, inverse = np.unique(prompt_ids, return_inverse=True)
    counts = np.bincount(inverse)
    if len(groups) != 96 or not np.all(counts == 18):
        raise ValueError("E40 train-fold prompt groups changed")
    means = np.zeros((len(groups), values.shape[1]), dtype=np.float64)
    np.add.at(means, inverse, values)
    means /= counts[:, None]
    residual = values - means[inverse]
    residual_sums = np.zeros_like(means)
    np.add.at(residual_sums, inverse, residual)
    if not np.isfinite(residual).all() or not np.allclose(
            residual_sums, 0.0, atol=1e-10, rtol=0):
        raise ValueError("E40 residual feature checks failed")
    return residual


def fit_style_folds(dev: pd.DataFrame, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    matrix = extract_features(dev["text"])
    if matrix.shape != (len(dev), 57) or not np.isfinite(matrix).all():
        raise ValueError("E40 57-feature matrix is invalid")
    labels = dev["model"].to_numpy()
    prompts = dev["prompt_id"].to_numpy(dtype=str)
    ordinary = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float64)
    residual = np.full_like(ordinary, np.nan)
    resources = []
    for fold in FOLDS:
        start = time.monotonic()
        fit = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        scaler = StandardScaler().fit(matrix[fit])
        x_train = scaler.transform(matrix[fit])
        x_valid = scaler.transform(matrix[valid])
        if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
            raise ValueError(f"Fold {fold} scaled style values are non-finite")
        ordinary_model = classifier().fit(x_train, labels[fit])
        residual_train = residualize_train(x_train, prompts[fit])
        residual_model = classifier().fit(residual_train, labels[fit])
        ordinary[valid] = ordered_scores(ordinary_model, x_valid)
        # The held-out prompt ID is deliberately not used in this inference.
        residual[valid] = ordered_scores(residual_model, x_valid)
        resources.append({"fold": fold, "train_rows": len(fit), "validation_rows": len(valid),
                          "train_prompts": len(np.unique(prompts[fit])),
                          "validation_prompts": len(np.unique(prompts[valid])),
                          "scaler_fitted_on_outer_training_only": True,
                          "validation_prompt_mean_subtracted": False,
                          "elapsed_seconds": time.monotonic() - start})
    if not np.isfinite(ordinary).all() or not np.isfinite(residual).all():
        raise ValueError("E40 fivefold style scores are incomplete")
    return ordinary, residual, resources


def per_class_recall(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {label: float(np.mean(predicted[truth == label] == label)) for label in LABELS}


def evaluate(dev: pd.DataFrame, folds: np.ndarray, checks: dict,
             e24_scores: np.ndarray, e24_pred: np.ndarray,
             ordinary: np.ndarray, residual: np.ndarray,
             resources: list[dict], elapsed_seconds: float) -> tuple[pd.DataFrame, dict]:
    truth = dev["model"].to_numpy()
    labels = np.asarray(LABELS)
    ordinary_pred = labels[ordinary.argmax(axis=1)]
    residual_pred = labels[residual.argmax(axis=1)]
    candidate_scores = e24_scores + FUSION_WEIGHT * (
        row_standardize(residual) - row_standardize(ordinary))
    candidate_pred = labels[candidate_scores.argmax(axis=1)]
    old_ba = float(balanced_accuracy_score(truth, e24_pred))
    new_ba = float(balanced_accuracy_score(truth, candidate_pred))
    if not np.isclose(old_ba, 0.425, atol=1e-12, rtol=0):
        raise ValueError("Archived E24 fixed fusion did not reproduce 0.425")
    old_recall = per_class_recall(truth, e24_pred)
    new_recall = per_class_recall(truth, candidate_pred)
    class_rows = [{"label": label, "e24_recall": old_recall[label],
                   "candidate_recall": new_recall[label],
                   "delta": new_recall[label] - old_recall[label]} for label in LABELS]
    fold_rows = []
    for fold in FOLDS:
        mask = folds == fold
        before = float(balanced_accuracy_score(truth[mask], e24_pred[mask]))
        after = float(balanced_accuracy_score(truth[mask], candidate_pred[mask]))
        fold_rows.append({"fold": fold, "e24_ba": before, "candidate_ba": after,
                          "delta": after - before,
                          "ordinary_style_ba": float(balanced_accuracy_score(truth[mask], ordinary_pred[mask])),
                          "residual_style_ba": float(balanced_accuracy_score(truth[mask], residual_pred[mask]))})
    non_decreasing = sum(row["delta"] >= -1e-12 for row in fold_rows)
    worst_class_drop = min(row["delta"] for row in class_rows)
    gate = {
        "ba_delta_at_least_0_01": new_ba - old_ba >= 0.01 - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": non_decreasing >= 4,
        "every_class_drop_strictly_less_than_0_05": worst_class_drop > -0.05 + 1e-12,
    }
    report = {
        "experiment": "e40_prompt_residual_style",
        "completed_utc": utc_now(),
        "preflight": checks,
        "protocol": {"features": "run_style.py original 57", "scaler": "StandardScaler train_fold_only",
                     "residual": "subtract train_prompt_mean after scaling; raw scaled validation",
                     "classifier": "LinearSVC C=1 class_weight=balanced dual=False max_iter=10000 seed=42",
                     "fusion": "e24 + 0.5*(row_z(residual)-row_z(ordinary))"},
        "development_rows": len(dev), "development_prompts": int(dev["prompt_id"].nunique()),
        "score": {"e24": old_ba,
                  "ordinary_style_single": float(balanced_accuracy_score(truth, ordinary_pred)),
                  "residual_style_single": float(balanced_accuracy_score(truth, residual_pred)),
                  "candidate_fixed": new_ba, "delta_vs_e24": new_ba - old_ba},
        "folds": fold_rows, "classes": class_rows,
        "non_decreasing_folds": non_decreasing,
        "worst_class_recall_delta": worst_class_drop,
        "paired_prompt_bootstrap": paired_prompt_bootstrap(
            truth, dev["prompt_id"].to_numpy(), e24_pred, candidate_pred,
            repeats=BOOTSTRAP_REPEATS, seed=SEED),
        "resource_folds": resources, "elapsed_seconds_total": elapsed_seconds,
        "gate": gate, "go": all(gate.values()),
        "holdout_or_test_scored": False,
        "per_row_oof_in_git": False,
    }
    frame = dev.loc[:, ["id", "prompt_id"]].copy()
    frame["true_model"] = truth
    frame["fold"] = folds
    for i, label in enumerate(LABELS):
        frame[f"e24_score_{label}"] = e24_scores[:, i]
        frame[f"ordinary_score_{label}"] = ordinary[:, i]
        frame[f"residual_score_{label}"] = residual[:, i]
        frame[f"candidate_score_{label}"] = candidate_scores[:, i]
    frame["e24_pred"] = e24_pred
    frame["ordinary_pred"] = ordinary_pred
    frame["residual_pred"] = residual_pred
    frame["candidate_pred"] = candidate_pred
    return frame, report


def run() -> None:
    dev, folds, checks = load_development()
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError(f"E40 output directory is already nonempty: {OUT}")
    # Keep E24 score access after the official data/protocol preflight.
    from run_e37_char4_likelihood import reproduce_e24, source_hashes

    source_hashes()
    e24_scores, e24_pred = reproduce_e24(dev, folds)
    write_new_json(OUT / "run_started.json", {"started_utc": utc_now(), "preflight": checks,
                                              "e24_fixed_reproduced": True})
    started = time.monotonic()
    ordinary, residual, resources = fit_style_folds(dev, folds)
    frame, report = evaluate(dev, folds, checks, e24_scores, e24_pred,
                             ordinary, residual, resources, time.monotonic() - started)
    with (OUT / "oof_scores.csv").open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    report["oof_scores_sha256"] = sha256(OUT / "oof_scores.csv")
    write_new_json(OUT / "metrics.json", report)
    print(json.dumps({"metrics": str(OUT / "metrics.json"), "go": report["go"]},
                     ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.command == "preflight":
        _, _, checks = load_development()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return
    run()


if __name__ == "__main__":
    main()
