#!/usr/bin/env python3
"""Independent read-only audit of completed, locked E37 fivefold OOF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/e37_char4_likelihood"
E24 = ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv"
E24_SCORES = {
    "sparse": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
    "roberta": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
    "modern_42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
    "modern_17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
    "modern_73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
}
KEYS = ["id", "prompt_id", "true_model", "fold"]
SCORE_COLS = [f"score_{label}" for label in LABELS]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_z(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise AssertionError("Row score standard deviation is zero or nonfinite")
    return (x - mean) / sd


def rec(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {label: float((pred[truth == label] == label).mean()) for label in LABELS}


def main() -> None:
    lock_path, oof_path, metrics_path = [OUT / name for name in
                                         ("protocol_lock.json", "oof_scores.csv", "metrics.json")]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    fp = lock["fingerprint"]
    assert metrics["protocol_lock_sha256"] == digest(lock_path)
    assert metrics["oof_scores_sha256"] == digest(oof_path)
    for name, relative in {
        "script": "run_e37_char4_likelihood.py",
        "config": "configs/e37_char4_conditional_likelihood.json",
        "preregistration": "docs/e37_char4_likelihood_preregistration.md",
        "run_e0": "run_e0.py", "run_fixed_blend": "run_fixed_blend.py",
        "run_score_normalization": "run_score_normalization.py",
    }.items():
        assert fp["code_and_protocol_sha256"][name] == digest(ROOT / relative), name
    for name, path in E24_SCORES.items():
        locked_name = {"sparse": "e24_sparse", "roberta": "roberta"}.get(name, name)
        assert digest(path) == fp["e24_source_sha256"][locked_name]
    assert digest(E24) == fp["e24_saved_oof_sha256"]
    oof = pd.read_csv(oof_path, dtype={key: str for key in KEYS})
    old = pd.read_csv(E24, dtype={key: str for key in KEYS})
    assert len(oof) == 2160 and oof["id"].is_unique
    assert oof[KEYS].equals(old[KEYS])
    assert oof["prompt_id"].nunique() == 120
    assert set(oof["fold"]) == {str(i) for i in range(1, 6)}
    assert set(oof["true_model"]) == set(LABELS)
    assert all((oof["true_model"] == label).sum() == 120 for label in LABELS)
    for fold in range(1, 6):
        subset = oof[oof["fold"] == str(fold)]
        assert len(subset) == 432 and subset["prompt_id"].nunique() == 24
        assert all((subset["true_model"] == label).sum() == 24 for label in LABELS)
    assert (oof.groupby("prompt_id")["fold"].nunique() == 1).all()
    labels = np.asarray(LABELS)
    truth = oof["true_model"].to_numpy()
    main = oof[SCORE_COLS].to_numpy(dtype=np.float64)
    mean = oof[[f"mean_score_{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    assert np.isfinite(main).all() and np.isfinite(mean).all()
    assert np.array_equal(labels[main.argmax(axis=1)], oof["pred_e37_main"].to_numpy())
    assert np.array_equal(labels[mean.argmax(axis=1)], oof["pred_e37_mean_diagnostic"].to_numpy())
    streams = {}
    for name, path in E24_SCORES.items():
        frame = pd.read_csv(path, dtype={key: str for key in KEYS})
        assert frame[KEYS].equals(oof[KEYS]), name
        streams[name] = frame[SCORE_COLS].to_numpy(dtype=np.float64)
    old_scores = (row_z(streams["sparse"]) + row_z(streams["roberta"])
                  + np.mean(np.stack([row_z(streams[f"modern_{seed}"])
                                      for seed in (42, 17, 73)]), axis=0))
    old_pred = labels[old_scores.argmax(axis=1)]
    assert np.array_equal(old_pred, old["pred_e16_with_e24"].to_numpy())
    new_pred = labels[(old_scores + row_z(main)).argmax(axis=1)]
    assert np.array_equal(new_pred, oof["pred_e24_plus_e37_main"].to_numpy())
    scores = metrics["score"]
    for name, pred in (("e24_fixed", old_pred), ("e37_main_single", labels[main.argmax(axis=1)]),
                       ("e37_mean_single_diagnostic", labels[mean.argmax(axis=1)]),
                       ("e24_plus_e37_main_fixed", new_pred)):
        assert np.isclose(balanced_accuracy_score(truth, pred), scores[name], atol=1e-12, rtol=0), name
    old_rec, new_rec = rec(truth, old_pred), rec(truth, new_pred)
    for row in metrics["class_recall"]:
        label = row["model"]
        assert np.isclose(row["e24_recall"], old_rec[label], atol=1e-12)
        assert np.isclose(row["e37_fusion_recall"], new_rec[label], atol=1e-12)
    for row in metrics["fold_metrics"]:
        mask = oof["fold"].eq(str(row["fold"])).to_numpy()
        assert np.isclose(balanced_accuracy_score(truth[mask], old_pred[mask]),
                          row["e24_balanced_accuracy"], atol=1e-12)
        assert np.isclose(balanced_accuracy_score(truth[mask], new_pred[mask]),
                          row["e37_fusion_balanced_accuracy"], atol=1e-12)
    resources = metrics["resources"]
    assert resources["gpu_used"] is False and resources["resource_gate_passed"] is True
    assert resources["peak_rss_bytes"] < 4 * 2**30
    assert resources["elapsed_seconds_total"] < 3600
    assert all(row["elapsed_seconds"] < 900 for row in metrics["fold_resource"])
    assert metrics["mechanism_preflight_gate_passed"] is False
    assert metrics["holdout_or_test_scored"] is False
    report = {
        "audit_passed": True, "protocol_lock_sha256": digest(lock_path),
        "oof_scores_sha256": digest(oof_path), "metrics_sha256": digest(metrics_path),
        "rows": len(oof), "prompts": oof["prompt_id"].nunique(),
        "folds": 5, "all_e24_predictions_reproduced": True,
        "all_saved_e37_predictions_reproduced": True,
        "all_fold_and_class_metrics_reproduced": True,
        "resources_within_locked_limits": True,
        "mechanism_preflight_gate_passed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    path = OUT / "independent_audit.json"
    if path.exists():
        raise ValueError("Audit file already exists")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
