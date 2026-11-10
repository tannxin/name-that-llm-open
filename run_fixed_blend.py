#!/usr/bin/env python3
"""Equal-rank blend of two aligned, prompt-grouped OOF score files.

This is a development-set diagnostic. It never reads the locked holdout.
The left model wins exact rank ties; no weight is tuned on OOF labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS


KEY_COLUMNS = ("id", "prompt_id", "true_model", "fold")
SCORE_COLUMNS = tuple(f"score_{label}" for label in LABELS)


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={key: str for key in KEY_COLUMNS})
    missing = sorted(set(KEY_COLUMNS + SCORE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame["id"].duplicated().any() or frame.empty:
        raise ValueError(f"{path} has duplicate IDs or no rows")
    scores = frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{path} has non-finite decision scores")
    return frame


def ranks(scores: np.ndarray) -> np.ndarray:
    # 0 is the lowest score, K-1 is the highest; score scale cancels.
    return np.argsort(np.argsort(scores, axis=1), axis=1).astype(np.float32)


def run(left_path: Path, right_path: Path, output_dir: Path) -> None:
    left = load_scores(left_path)
    right = load_scores(right_path)
    if len(left) != 2160 or left["prompt_id"].nunique() != 120:
        raise ValueError("Expected exactly the saved 2,160-row development partition")
    for key in KEY_COLUMNS:
        if not left[key].equals(right[key]):
            raise ValueError(f"Score files differ in {key}")
    if set(left["true_model"]) != set(LABELS):
        raise ValueError("Unexpected true label set")
    if set(left["fold"]) != {str(value) for value in range(1, 6)}:
        raise ValueError("Unexpected fold assignments")
    labels = left["true_model"].to_numpy()
    left_scores = left.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    right_scores = right.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    left_ranks = ranks(left_scores)
    right_ranks = ranks(right_scores)
    summed = left_ranks + right_ranks
    # Break equal-rank sums in favor of the left model without using labels.
    selected = (summed + left_ranks * 1e-4).argmax(axis=1)
    predicted = np.asarray(LABELS)[selected]
    left_predicted = np.asarray(LABELS)[left_scores.argmax(axis=1)]
    right_predicted = np.asarray(LABELS)[right_scores.argmax(axis=1)]
    scores = {
        "left_balanced_accuracy": float(balanced_accuracy_score(labels, left_predicted)),
        "right_balanced_accuracy": float(balanced_accuracy_score(labels, right_predicted)),
        "equal_rank_blend_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }
    fold_scores = []
    for fold in range(1, 6):
        mask = left["fold"].eq(str(fold)).to_numpy()
        fold_scores.append({
            "fold": fold,
            "left_balanced_accuracy": float(balanced_accuracy_score(labels[mask], left_predicted[mask])),
            "right_balanced_accuracy": float(balanced_accuracy_score(labels[mask], right_predicted[mask])),
            "blend_balanced_accuracy": float(balanced_accuracy_score(labels[mask], predicted[mask])),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = left.loc[:, list(KEY_COLUMNS)].copy()
    oof["pred_model"] = predicted
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(output_dir / "fold_metrics.csv", index=False)
    metadata = {
        "experiment": "e7_equal_rank_blend",
        "left_score_file": str(left_path.resolve()),
        "right_score_file": str(right_path.resolve()),
        "rule": "equal sum of per-row model-label ranks; left model breaks exact ties",
        "weights_tuned": False,
        "score_is_development_oof_not_final_test": True,
        "n_rows_development": len(left),
        "n_prompts_development": int(left["prompt_id"].nunique()),
        "n_folds": 5,
        **scores,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.left, args.right, args.output_dir)


if __name__ == "__main__":
    main()
