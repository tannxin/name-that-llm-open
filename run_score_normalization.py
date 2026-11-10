#!/usr/bin/env python3
"""Compare fixed rank and row-standardized score fusion on grouped OOF rows.

No fusion weight, temperature, or other parameter is selected from labels.
The reserved 30-prompt holdout is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores, ranks


def row_standardize(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if (scale <= 0).any():
        raise ValueError("A row has identical class scores")
    return centered / scale


def paired_prompt_bootstrap(
    true: np.ndarray, prompts: np.ndarray, old: np.ndarray, new: np.ndarray,
    repeats: int = 20_000, seed: int = 42,
) -> dict:
    unique_prompts, group = np.unique(prompts, return_inverse=True)
    old_hits = np.zeros((len(unique_prompts), len(LABELS)), dtype=np.int16)
    new_hits = np.zeros_like(old_hits)
    class_totals = np.zeros_like(old_hits)
    for label_index, label in enumerate(LABELS):
        mask = true == label
        np.add.at(class_totals[:, label_index], group[mask], 1)
        np.add.at(old_hits[:, label_index], group[mask & (old == label)], 1)
        np.add.at(new_hits[:, label_index], group[mask & (new == label)], 1)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 500):
        stop = min(start + 500, repeats)
        choices = rng.integers(0, len(unique_prompts), size=(stop - start, len(unique_prompts)))
        denominator = class_totals[choices].sum(axis=1)
        if (denominator == 0).any():
            raise ValueError("A bootstrap draw omitted a class")
        old_score = (old_hits[choices].sum(axis=1) / denominator).mean(axis=1)
        new_score = (new_hits[choices].sum(axis=1) / denominator).mean(axis=1)
        differences[start:stop] = new_score - old_score
    return {
        "difference": float(balanced_accuracy_score(true, new) - balanced_accuracy_score(true, old)),
        "confidence_interval_95": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "bootstrap_unit": "prompt_id",
        "repeats": repeats,
        "seed": seed,
    }


def run(left_path: Path, right_path: Path, reference_path: Path, output_dir: Path) -> None:
    left, right = load_scores(left_path), load_scores(right_path)
    if len(left) != 2160 or left["prompt_id"].nunique() != 120:
        raise ValueError("Expected the fixed 2,160-row development partition")
    for key in KEY_COLUMNS:
        if not left[key].equals(right[key]):
            raise ValueError(f"The OOF score files differ in {key}")
    if set(left["fold"]) != {str(i) for i in range(1, 6)}:
        raise ValueError("The OOF fold assignments differ from the five-fold protocol")
    reference = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    for key in KEY_COLUMNS:
        if not left[key].equals(reference[key]):
            raise ValueError(f"The saved reference differs in {key}")
    if not set(reference["pred_model"]).issubset(LABELS):
        raise ValueError("The saved reference has an unknown label")

    left_scores = left.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    right_scores = right.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    labels = left["true_model"].to_numpy()
    left_ranks = ranks(left_scores)
    rank_pred = np.asarray(LABELS)[(left_ranks + ranks(right_scores) + left_ranks * 1e-4).argmax(axis=1)]
    if not np.array_equal(rank_pred, reference["pred_model"].to_numpy()):
        raise ValueError("Fixed rank fusion did not reproduce the published E7e-R OOF")
    z_scores = row_standardize(left_scores) + row_standardize(right_scores)
    z_pred = np.asarray(LABELS)[z_scores.argmax(axis=1)]
    fold_scores = []
    for fold in range(1, 6):
        mask = left["fold"].eq(str(fold)).to_numpy()
        fold_scores.append({
            "fold": fold,
            "rank_balanced_accuracy": float(balanced_accuracy_score(labels[mask], rank_pred[mask])),
            "zscore_balanced_accuracy": float(balanced_accuracy_score(labels[mask], z_pred[mask])),
        })
    rank_confusion = confusion_matrix(labels, rank_pred, labels=LABELS)
    z_confusion = confusion_matrix(labels, z_pred, labels=LABELS)
    class_recall = pd.DataFrame({
        "model": LABELS,
        "support": rank_confusion.sum(axis=1),
        "rank_recall": rank_confusion.diagonal() / rank_confusion.sum(axis=1),
        "zscore_recall": z_confusion.diagonal() / z_confusion.sum(axis=1),
    })
    class_recall["difference"] = class_recall["zscore_recall"] - class_recall["rank_recall"]
    metrics = {
        "experiment": "e10_fixed_fusion_normalization",
        "rank_balanced_accuracy": float(balanced_accuracy_score(labels, rank_pred)),
        "zscore_balanced_accuracy": float(balanced_accuracy_score(labels, z_pred)),
        "paired_prompt_bootstrap_zscore_minus_rank": paired_prompt_bootstrap(
            labels, left["prompt_id"].to_numpy(), rank_pred, z_pred
        ),
        "fold_scores": fold_scores,
        "n_rows_development": len(left),
        "n_prompts_development": int(left["prompt_id"].nunique()),
        "reference_exactly_reproduced": True,
        "calibration_fitted": False,
        "holdout_evaluated": False,
        "left_sha256": sha256(left_path),
        "right_sha256": sha256(right_path),
        "reference_sha256": sha256(reference_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(fold_scores).to_csv(output_dir / "fold_metrics.csv", index=False)
    class_recall.to_csv(output_dir / "class_metrics.csv", index=False)
    oof = left.loc[:, list(KEY_COLUMNS)].copy()
    oof["rank_pred_model"] = rank_pred
    oof["zscore_pred_model"] = z_pred
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    parser.add_argument("--right", type=Path, default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/e7_blend_e0c_roberta_ft_scalerfix/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e10_fusion_normalization"))
    args = parser.parse_args()
    run(args.left, args.right, args.reference, args.output_dir)


if __name__ == "__main__":
    main()
