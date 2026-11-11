#!/usr/bin/env python3
"""E37 CPU-only, prospectively locked character likelihood mechanism probe.

``lock`` hashes the protocol and checks the official development partition
without fitting or reading E37 validation scores. ``run`` fits all five fixed
prompt folds once, then reports OOF metrics. No GPU or network is used.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/e37_char4_conditional_likelihood.json"
DOC_PATH = ROOT / "docs/e37_char4_likelihood_preregistration.md"
OUT = ROOT / "outputs/e37_char4_likelihood"
LOCK_PATH = OUT / "protocol_lock.json"
TRAIN_PATH = ROOT / "data/train.csv"
E0_DIR = ROOT / "outputs/e0_tfidf_svm"
E24_DIR = ROOT / "outputs/e24_prvsiyan_analysis"
EXPECTED_TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_E0_MANIFEST_SHA = "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15"
EXPECTED_E0_OOF_SHA = "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa"
EXPECTED_E24_METRICS_SHA = "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557"
EXPECTED_E24_OOF_SHA = "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448"
E24_SOURCES = {
    "e24_sparse": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
    "roberta": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
    "modern_42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
    "modern_17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
    "modern_73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
}
EXPECTED_E24_SOURCE_SHA = {
    "e24_sparse": "845c657f03b761f7b8f62fc17a2da583b6ecfd5962d5be1ae6fe485e5a9c3575",
    "roberta": "86c6621058dc630b690bc21a58b5767872466c416395c02c327491b8b16c80cd",
    "modern_42": "0ebf57806393d5175842ae66e2c18080f0c536f40a26e06e3063fc31097ce233",
    "modern_17": "da391ea077c200e9735237183cf02f7d57b5b809685b437d37b5e0ef3346c189",
    "modern_73": "23de22abb848effb851fd2f079283d6c726a2d8f9c339920a227d4b981989ba8",
}
FOLDS = (1, 2, 3, 4, 5)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def config() -> dict:
    value = read_json(CONFIG_PATH)
    required = {
        "experiment": "e37_char4_conditional_likelihood",
        "protocol_version": 1,
        "seed": 42,
        "ngram_length": 4,
        "max_features": 100000,
        "alpha": 1.0,
        "temperatures": [1.0, 2.0, 4.0, 8.0],
        "primary_readout": "mean_over_temperatures_of_mean_over_four_character_occurrences_of_clipped_relative_likelihood",
        "secondary_readout": "mean_log_likelihood",
        "fixed_e24_fusion_coefficient": 1.0,
        "minimum_fusion_balanced_accuracy_delta": 0.01,
        "minimum_non_decreasing_folds": 4,
        "maximum_allowed_class_recall_drop_exclusive": 0.05,
        "bootstrap_prompt_repeats": 20000,
        "max_text_characters": 10000,
        "max_peak_rss_gib": 4.0,
        "max_seconds_per_fold": 900,
        "max_seconds_total": 3600,
    }
    if value != required:
        raise ValueError("E37 configuration differs from the registered protocol")
    return value


def source_hashes() -> dict[str, str]:
    if sha256(E24_DIR / "metrics.json") != EXPECTED_E24_METRICS_SHA:
        raise ValueError("E24 metrics hash changed")
    for name, path in E24_SOURCES.items():
        if sha256(path) != EXPECTED_E24_SOURCE_SHA[name]:
            raise ValueError(f"E24 archived source hash changed: {name}")
    if sha256(E24_DIR / "oof_predictions.csv") != EXPECTED_E24_OOF_SHA:
        raise ValueError("E24 saved OOF hash changed")
    return EXPECTED_E24_SOURCE_SHA.copy()


def fingerprint() -> dict:
    paths = {
        "script": Path(__file__).resolve(),
        "config": CONFIG_PATH,
        "preregistration": DOC_PATH,
        "run_e0": ROOT / "run_e0.py",
        "run_fixed_blend": ROOT / "run_fixed_blend.py",
        "run_score_normalization": ROOT / "run_score_normalization.py",
    }
    if sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA:
        raise ValueError("Official train.csv hash changed")
    if sha256(E0_DIR / "split_manifest.csv") != EXPECTED_E0_MANIFEST_SHA:
        raise ValueError("E0 split manifest hash changed")
    if sha256(E0_DIR / "oof_predictions.csv") != EXPECTED_E0_OOF_SHA:
        raise ValueError("E0 OOF hash changed")
    return {
        "code_and_protocol_sha256": {name: sha256(path) for name, path in paths.items()},
        "official_train_sha256": EXPECTED_TRAIN_SHA,
        "e0_split_manifest_sha256": EXPECTED_E0_MANIFEST_SHA,
        "e0_oof_sha256": EXPECTED_E0_OOF_SHA,
        "e24_metrics_sha256": EXPECTED_E24_METRICS_SHA,
        "e24_saved_oof_sha256": EXPECTED_E24_OOF_SHA,
        "e24_source_sha256": source_hashes(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }


def load_development() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Use holdout prompt IDs only for routing; never retain holdout text."""
    manifest = pd.read_csv(E0_DIR / "split_manifest.csv", dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"] or manifest["prompt_id"].duplicated().any():
        raise ValueError("E0 manifest columns or IDs changed")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("E0 manifest split labels changed")
    dev_prompts = set(manifest.loc[manifest["split"].eq("development"), "prompt_id"])
    if len(manifest) != 150 or len(dev_prompts) != 120 or manifest["split"].eq("holdout").sum() != 30:
        raise ValueError("E0 development or holdout prompt counts changed")
    rows = []
    with TRAIN_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"id", "prompt_id", "text", "model"}:
            raise ValueError("Official train.csv columns changed")
        for row in reader:
            if row["prompt_id"] in dev_prompts:
                rows.append({key: row[key] for key in ("id", "prompt_id", "text", "model")})
    dev = pd.DataFrame(rows)
    base = pd.read_csv(E0_DIR / "oof_predictions.csv", dtype=str, keep_default_na=False,
                       usecols=["id", "prompt_id", "true_model", "fold"])
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120 or dev["id"].duplicated().any():
        raise ValueError("E37 development dimensions changed")
    if not expected.reset_index(drop=True).equals(base.loc[:, ["id", "prompt_id", "true_model"]]):
        raise ValueError("E0 OOF identity differs from official development rows")
    folds = pd.to_numeric(base["fold"], errors="raise").to_numpy(dtype=np.int64)
    if set(folds) != set(FOLDS) or set(dev["model"]) != set(LABELS):
        raise ValueError("E0 folds or labels changed")
    for fold in FOLDS:
        fit = dev.loc[folds != fold]
        valid = dev.loc[folds == fold]
        if (len(fit) != 1728 or len(valid) != 432 or fit["prompt_id"].nunique() != 96
                or valid["prompt_id"].nunique() != 24
                or set(fit["prompt_id"]) & set(valid["prompt_id"])
                or any((fit["model"] == label).sum() != 96 for label in LABELS)
                or any((valid["model"] == label).sum() != 24 for label in LABELS)):
            raise ValueError(f"E0 fold {fold} dimensions, class balance, or prompt isolation changed")
    lengths = dev["text"].map(len).to_numpy()
    if np.min(lengths) < config()["ngram_length"] or np.max(lengths) > config()["max_text_characters"]:
        raise ValueError("Development text violates fixed character length bounds")
    checks = {
        "development_rows": len(dev), "development_prompts": dev["prompt_id"].nunique(),
        "holdout_prompts_excluded": 30, "folds": len(FOLDS),
        "min_development_characters": int(lengths.min()),
        "max_development_characters": int(lengths.max()),
        "total_development_fourgram_occurrences": int(np.sum(lengths - 3)),
        "holdout_text_loaded": False,
    }
    return dev, folds, checks


def peak_rss_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_t), ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t), ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t), ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t), ("PeakPagefileUsage", size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("Psapi.dll")
        kernel = ctypes.WinDLL("Kernel32.dll")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise RuntimeError("Unable to read Windows process peak RSS")
        return int(counters.PeakWorkingSetSize)
    if sys.platform.startswith("linux"):
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    raise RuntimeError("E37 resource measurement supports only Windows and Linux")


def resource_check(start: float, fold_start: float | None = None) -> None:
    cfg = config()
    if peak_rss_bytes() > cfg["max_peak_rss_gib"] * 2**30:
        raise RuntimeError("E37 peak RSS exceeded locked 4 GiB cap")
    if time.monotonic() - start > cfg["max_seconds_total"]:
        raise RuntimeError("E37 elapsed time exceeded locked 3600-second cap")
    if fold_start is not None and time.monotonic() - fold_start > cfg["max_seconds_per_fold"]:
        raise RuntimeError("E37 fold elapsed time exceeded locked 900-second cap")


def lock() -> None:
    if OUT.exists():
        raise ValueError("E37 output directory already exists; refusing another lock")
    cfg = config()
    fp = fingerprint()  # SHA only for E24 score files, no score values read.
    _, _, checks = load_development()  # No likelihood model fitted or scored.
    resource_check(time.monotonic())
    value = {
        "experiment": cfg["experiment"], "protocol_version": cfg["protocol_version"],
        "stage": "prospective_cpu_lock_before_any_e37_validation_scores",
        "locked_at_utc": utc_now(), "fingerprint": fp, "preflight": checks,
        "e37_validation_scores_read": False, "old_e24_score_values_read_at_lock": False,
        "gpu_used": False, "holdout_or_test_scored": False,
    }
    write_new_json(LOCK_PATH, value)
    print(json.dumps({"stage": "lock", "protocol_lock_sha256": sha256(LOCK_PATH),
                      "preflight": checks, "e37_validation_scores_read": False}, indent=2))


def checked_lock() -> dict:
    locked = read_json(LOCK_PATH)
    cfg = config()
    if (locked.get("experiment") != cfg["experiment"]
            or locked.get("protocol_version") != cfg["protocol_version"]
            or locked.get("stage") != "prospective_cpu_lock_before_any_e37_validation_scores"
            or locked.get("e37_validation_scores_read") is not False
            or locked.get("gpu_used") is not False
            or locked.get("fingerprint") != fingerprint()):
        raise ValueError("E37 code, config, data, runtime, or sources differ from prospective lock")
    return locked


def char4_analyzer(text: str):
    return (text[index:index + 4] for index in range(len(text) - 3))


def relative_readout(log_likelihood: np.ndarray, weights: np.ndarray,
                     temperatures: tuple[float, ...]) -> np.ndarray:
    """Exact ATLAS clipped evidence formula, weighted over repeated 4-grams."""
    if log_likelihood.ndim != 2 or log_likelihood.shape[1] != len(LABELS):
        raise ValueError("Expected a [unique_4gram, 18] log-likelihood matrix")
    if len(weights) != len(log_likelihood) or np.any(weights < 0):
        raise ValueError("Fourgram multiplicities are invalid")
    positive = np.maximum(log_likelihood[:, None, :] - log_likelihood[:, :, None], 0.0)
    penalty = positive.sum(axis=2)
    return sum(weights @ np.exp(-penalty / temperature) for temperature in temperatures) / len(temperatures)


def fit_and_score_fold(dev: pd.DataFrame, folds: np.ndarray, fold: int, start: float,
                       cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    fold_start = time.monotonic()
    fit_mask = folds != fold
    valid_mask = folds == fold
    fit_texts = dev.loc[fit_mask, "text"].tolist()
    fit_labels = dev.loc[fit_mask, "model"].to_numpy()
    valid_texts = dev.loc[valid_mask, "text"].tolist()
    vectorizer = CountVectorizer(analyzer=char4_analyzer, lowercase=False,
                                 max_features=cfg["max_features"], dtype=np.int32)
    x_train = vectorizer.fit_transform(fit_texts).tocsr()
    vocabulary_size = x_train.shape[1]
    if vocabulary_size < 1 or vocabulary_size > cfg["max_features"]:
        raise ValueError(f"Fold {fold} vocabulary outside locked range")
    class_counts = np.vstack([
        np.asarray(x_train[fit_labels == label].sum(axis=0)).ravel().astype(np.float64)
        for label in LABELS
    ])
    class_totals = np.asarray([
        sum(len(text) - 3 for text, label_value in zip(fit_texts, fit_labels) if label_value == label)
        for label in LABELS
    ], dtype=np.float64)
    known_totals = class_counts.sum(axis=1)
    oov_counts = class_totals - known_totals
    if np.any(oov_counts < 0) or np.any(class_totals <= 0):
        raise ValueError(f"Fold {fold} invalid class or OOV counts")
    denominator = class_totals[:, None] + vocabulary_size + 1
    log_known = np.log((class_counts + cfg["alpha"]) / denominator)
    log_oov = np.log((oov_counts + cfg["alpha"]) / denominator[:, 0])
    if not np.allclose(np.exp(log_known).sum(axis=1) + np.exp(log_oov), 1, atol=1e-10, rtol=0):
        raise ValueError(f"Fold {fold} class likelihood probabilities do not normalize")
    resource_check(start, fold_start)
    x_valid = vectorizer.transform(valid_texts).tocsr()
    primary = np.empty((len(valid_texts), len(LABELS)), dtype=np.float64)
    secondary = np.empty_like(primary)
    total_oov = 0
    total_occurrences = 0
    temperatures = tuple(cfg["temperatures"])
    for row, text in enumerate(valid_texts):
        a, b = x_valid.indptr[row], x_valid.indptr[row + 1]
        columns = x_valid.indices[a:b]
        counts = x_valid.data[a:b].astype(np.float64)
        total = len(text) - 3
        oov = total - int(counts.sum())
        if total <= 0 or oov < 0:
            raise ValueError(f"Fold {fold} validation fourgram count is invalid")
        token_ll = log_known[:, columns].T
        evidence_sum = relative_readout(token_ll, counts, temperatures)
        mean_ll_sum = counts @ token_ll
        if oov:
            evidence_sum += oov * relative_readout(log_oov[None, :], np.ones(1), temperatures)
            mean_ll_sum += oov * log_oov
        primary[row] = evidence_sum / total
        secondary[row] = mean_ll_sum / total
        total_oov += oov
        total_occurrences += total
        if row % 50 == 0:
            resource_check(start, fold_start)
    if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
        raise ValueError(f"Fold {fold} produced non-finite scores")
    resource_check(start, fold_start)
    metadata = {
        "fold": fold, "train_rows": len(fit_texts), "validation_rows": len(valid_texts),
        "vocabulary_size": vocabulary_size,
        "train_class_total_fourgram_min": int(class_totals.min()),
        "train_class_total_fourgram_max": int(class_totals.max()),
        "train_oov_occurrences": int(oov_counts.sum()),
        "validation_fourgram_occurrences": int(total_occurrences),
        "validation_oov_occurrences": int(total_oov),
        "validation_oov_fraction": float(total_oov / total_occurrences),
        "elapsed_seconds": time.monotonic() - fold_start,
        "peak_rss_bytes_after_fold": peak_rss_bytes(),
    }
    return primary, secondary, metadata


def aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"{name} row count differs from E24")
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} row identity differs in {key}")


def reproduce_e24(dev: pd.DataFrame, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    metrics = read_json(E24_DIR / "metrics.json")
    saved = pd.read_csv(E24_DIR / "oof_predictions.csv", dtype=str, keep_default_na=False)
    files = {name: load_scores(path) for name, path in E24_SOURCES.items()}
    reference = files["e24_sparse"]
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if not expected.equals(reference.loc[:, ["id", "prompt_id", "true_model"]]):
        raise ValueError("E24 score rows differ from official development rows")
    if not np.array_equal(folds.astype(str), reference["fold"].to_numpy()):
        raise ValueError("E24 fold IDs differ from E0")
    for name, frame in [*files.items(), ("saved_e24", saved)]:
        aligned(reference, frame, name)
    def mat(name: str) -> np.ndarray:
        return files[name].loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    original = (row_standardize(mat("e24_sparse")) + row_standardize(mat("roberta"))
                + np.mean(np.stack([row_standardize(mat(f"modern_{seed}"))
                                    for seed in (42, 17, 73)]), axis=0))
    labels = np.asarray(LABELS)
    predicted = labels[original.argmax(axis=1)]
    if (not np.array_equal(predicted, saved["pred_e16_with_e24"].to_numpy())
            or not np.isclose(balanced_accuracy_score(dev["model"], predicted),
                              metrics["score"]["e16_with_e24"], atol=1e-12, rtol=0)
            or not np.isclose(metrics["score"]["e16_with_e24"], 0.425, atol=1e-12, rtol=0)):
        raise ValueError("Archived E24 fixed fusion did not reproduce exactly")
    return original, predicted


def per_class_recall(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {label: float(np.mean(predicted[truth == label] == label)) for label in LABELS}


def evaluate(dev: pd.DataFrame, folds: np.ndarray, primary: np.ndarray,
             secondary: np.ndarray, fold_resources: list[dict], started: float) -> tuple[pd.DataFrame, dict]:
    e24_scores, e24_pred = reproduce_e24(dev, folds)
    truth = dev["model"].to_numpy()
    labels = np.asarray(LABELS)
    main_pred = labels[primary.argmax(axis=1)]
    mean_pred = labels[secondary.argmax(axis=1)]
    fusion_pred = labels[(e24_scores + row_standardize(primary)).argmax(axis=1)]
    old_ba = float(balanced_accuracy_score(truth, e24_pred))
    new_ba = float(balanced_accuracy_score(truth, fusion_pred))
    old_recall, new_recall = per_class_recall(truth, e24_pred), per_class_recall(truth, fusion_pred)
    class_rows = [{"model": label, "e24_recall": old_recall[label],
                   "e37_fusion_recall": new_recall[label],
                   "difference": new_recall[label] - old_recall[label]} for label in LABELS]
    fold_rows = []
    for fold in FOLDS:
        mask = folds == fold
        before = float(balanced_accuracy_score(truth[mask], e24_pred[mask]))
        after = float(balanced_accuracy_score(truth[mask], fusion_pred[mask]))
        fold_rows.append({"fold": fold, "e24_balanced_accuracy": before,
                          "e37_fusion_balanced_accuracy": after, "difference": after - before,
                          "e37_main_single_balanced_accuracy": float(balanced_accuracy_score(truth[mask], main_pred[mask]))})
    cfg = config()
    non_decrease = sum(row["difference"] >= -1e-12 for row in fold_rows)
    worst_class_delta = min(row["difference"] for row in class_rows)
    gate = {
        "fusion_delta_at_least_0_01": new_ba - old_ba >= cfg["minimum_fusion_balanced_accuracy_delta"] - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": non_decrease >= cfg["minimum_non_decreasing_folds"],
        "every_class_drop_strictly_less_than_0_05": worst_class_delta > -cfg["maximum_allowed_class_recall_drop_exclusive"] + 1e-12,
    }
    frame = dev.loc[:, ["id", "prompt_id"]].copy()
    frame["true_model"] = truth
    frame["fold"] = folds
    for index, label in enumerate(LABELS):
        frame[f"score_{label}"] = primary[:, index]
        frame[f"mean_score_{label}"] = secondary[:, index]
    frame["pred_e37_main"] = main_pred
    frame["pred_e37_mean_diagnostic"] = mean_pred
    frame["pred_e24_plus_e37_main"] = fusion_pred
    metrics = {
        "experiment": cfg["experiment"], "protocol_version": cfg["protocol_version"],
        "protocol_lock_sha256": sha256(LOCK_PATH), "completed_at_utc": utc_now(),
        "development_rows": len(dev), "development_prompts": dev["prompt_id"].nunique(),
        "labels": list(LABELS), "folds_complete": len(fold_rows),
        "model": "class_conditional_smoothed_four_character_multinomial_proxy_not_neural_ATLAS",
        "main_readout": cfg["primary_readout"], "secondary_readout_diagnostic_only": cfg["secondary_readout"],
        "e24_fixed_fusion_exactly_reproduced": True,
        "score": {"e24_fixed": old_ba,
                  "e37_main_single": float(balanced_accuracy_score(truth, main_pred)),
                  "e37_mean_single_diagnostic": float(balanced_accuracy_score(truth, mean_pred)),
                  "e24_plus_e37_main_fixed": new_ba,
                  "fusion_delta_vs_e24": new_ba - old_ba},
        "fold_metrics": fold_rows, "class_recall": class_rows,
        "fold_resource": fold_resources,
        "non_decreasing_folds": non_decrease,
        "worst_class_recall_delta": worst_class_delta,
        "paired_prompt_bootstrap": paired_prompt_bootstrap(
            truth, dev["prompt_id"].to_numpy(), e24_pred, fusion_pred,
            repeats=cfg["bootstrap_prompt_repeats"], seed=cfg["seed"]),
        "gate": gate, "mechanism_preflight_gate_passed": all(gate.values()),
        "resources": {"elapsed_seconds_total": time.monotonic() - started,
                      "peak_rss_bytes": peak_rss_bytes(),
                      "gpu_used": False, "resource_gate_passed": True},
        "holdout_or_test_scored": False,
        "full_atlas_or_kaggle_promotion": False,
    }
    return frame, metrics


def run() -> None:
    checked_lock()
    if (OUT / "run_started.json").exists() or (OUT / "oof_scores.csv").exists():
        raise ValueError("E37 already started; refusing a second run")
    dev, folds, checks = load_development()
    if checks != read_json(LOCK_PATH)["preflight"]:
        raise ValueError("E37 data preflight changed since lock")
    cfg = config()
    write_new_json(OUT / "run_started.json", {"stage": "fivefold_once", "started_at_utc": utc_now(),
                                               "protocol_lock_sha256": sha256(LOCK_PATH)})
    started = time.monotonic()
    primary = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float64)
    secondary = np.full_like(primary, np.nan)
    fold_resources = []
    try:
        resource_check(started)
        for fold in FOLDS:
            a, b, resource = fit_and_score_fold(dev, folds, fold, started, cfg)
            primary[folds == fold] = a
            secondary[folds == fold] = b
            fold_resources.append(resource)
            print(f"E37 fold {fold}/5 finished; metrics withheld until all folds complete", flush=True)
        if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
            raise ValueError("E37 fivefold OOF scores are incomplete")
        frame, metrics = evaluate(dev, folds, primary, secondary, fold_resources, started)
        resource_check(started)
        frame.to_csv(OUT / "oof_scores.csv", index=False)
        metrics["oof_scores_sha256"] = sha256(OUT / "oof_scores.csv")
        write_new_json(OUT / "metrics.json", metrics)
        print(json.dumps({"stage": "fivefold_complete", "protocol_lock_sha256": sha256(LOCK_PATH),
                          "oof_scores_sha256": metrics["oof_scores_sha256"],
                          "score": metrics["score"], "gate": metrics["gate"],
                          "resource_gate_passed": True}, indent=2), flush=True)
    except Exception as exc:
        if not (OUT / "failure.json").exists():
            write_new_json(OUT / "failure.json", {"failed_at_utc": utc_now(),
                                                   "reason": str(exc),
                                                   "folds_finished_without_displayed_scores": len(fold_resources),
                                                   "partial_validation_scores_saved": False})
        raise


def self_test() -> None:
    """Synthetic math/normalization checks; no official rows or scores read."""
    ll = np.zeros((1, len(LABELS)), dtype=np.float64)
    ll[0, 0] = 1.0
    weights = np.ones(1)
    score = relative_readout(ll, weights, (1.0, 2.0, 4.0, 8.0))
    if not np.isclose(score[0], 1.0, atol=1e-12) or not np.all(score[1:] < 1):
        raise AssertionError("ATLAS clipped score synthetic check failed")
    grams = list(char4_analyzer("Ab  C"))
    if grams != ["Ab  ", "b  C"]:
        raise AssertionError("Character analyzer normalized whitespace or case")
    print("E37 synthetic math and character analyzer checks passed; no validation scores read")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-test", "lock", "run"))
    stage = parser.parse_args().stage
    if stage == "self-test":
        self_test()
    elif stage == "lock":
        lock()
    else:
        run()


if __name__ == "__main__":
    main()
