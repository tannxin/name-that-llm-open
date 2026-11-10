#!/usr/bin/env python3
"""Recreate E0c OOF margins for later, leakage-safe score fusion.

Every feature extractor and classifier is refitted inside its E0 prompt fold.
The saved E0c OOF predictions must reproduce exactly before scores are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import load_e0_split


def run(train: Path, e0_dir: Path, e0c_dir: Path) -> None:
    frame = load_and_validate(train)
    dev, _, e0, folds, _ = load_e0_split(frame, e0_dir, train)
    expected = pd.read_csv(e0c_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for column in ("id", "prompt_id", "true_model", "fold"):
        if not expected[column].equals(e0[column]):
            raise ValueError(f"Saved E0c {column} differs from E0")
    e0c_metrics = json.loads((e0c_dir / "metrics.json").read_text(encoding="utf-8"))
    if e0c_metrics["train_sha256"] != sha256(train):
        raise ValueError("Saved E0c data hash differs from train.csv")
    labels = dev["model"].to_numpy()
    texts = dev["text"].to_numpy()
    predicted = np.empty(len(dev), dtype=object)
    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(dev.loc[fit_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompts")
        model = make_pipeline().fit(texts[fit_idx], labels[fit_idx])
        predicted[valid_idx] = model.predict(texts[valid_idx])
        class_order = model.named_steps["classifier"].classes_
        positions = [int(np.flatnonzero(class_order == label)[0]) for label in LABELS]
        scores[valid_idx] = model.decision_function(texts[valid_idx])[:, positions]
        print(f"completed fold {fold}/{N_FOLDS}", flush=True)
    if not np.array_equal(predicted, expected["pred_model"].to_numpy()):
        raise RuntimeError("Recomputed E0c OOF predictions differ from the saved run")
    if not np.isfinite(scores).all():
        raise RuntimeError("OOF decision scores are incomplete or non-finite")
    result = dev[["id", "prompt_id"]].copy()
    result["true_model"] = labels
    result["fold"] = folds
    for index, label in enumerate(LABELS):
        result[f"score_{label}"] = scores[:, index]
    result.to_csv(e0c_dir / "decision_scores.csv", index=False)
    print(f"Saved {(e0c_dir / 'decision_scores.csv').resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0c-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.e0c_dir)


if __name__ == "__main__":
    main()
