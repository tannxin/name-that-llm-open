#!/usr/bin/env python3
"""Small, fixed E0c hyperparameter grid on the saved development folds only.

The nine candidates vary only LinearSVC C and relative style block weight.
The selected OOF score is for model selection, not a new unbiased test score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import FEATURE_NAMES, load_e0_split


C_VALUES = (0.25, 1.0, 4.0)
STYLE_WEIGHTS = (0.5, 1.0, 2.0)


def weighted(matrix, weight: float):
    n_style = len(FEATURE_NAMES)
    return hstack((matrix[:, :-n_style], matrix[:, -n_style:] * weight), format="csr")


def run(train: Path, e0_dir: Path, e0c_dir: Path, output_dir: Path) -> None:
    frame = load_and_validate(train)
    dev, manifest, e0, folds, _ = load_e0_split(frame, e0_dir, train)
    e0c = pd.read_csv(e0c_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for column in ("id", "prompt_id", "true_model", "fold"):
        if not e0c[column].equals(e0[column]):
            raise ValueError(f"E0c {column} differs from E0")
    texts = dev["text"].to_numpy()
    labels = dev["model"].to_numpy()
    predictions = {
        (c, w): np.empty(len(dev), dtype=object)
        for c in C_VALUES for w in STYLE_WEIGHTS
    }
    fold_scores = {key: [] for key in predictions}
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks an official class")
        if set(dev.loc[fit_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompts")
        features = make_pipeline().named_steps["features"]
        fit_matrix = features.fit_transform(texts[fit_idx])
        valid_matrix = features.transform(texts[valid_idx])
        for weight in STYLE_WEIGHTS:
            fit_weighted = weighted(fit_matrix, weight)
            valid_weighted = weighted(valid_matrix, weight)
            for c in C_VALUES:
                head = LinearSVC(
                    C=c, class_weight="balanced", dual=True,
                    max_iter=10_000, random_state=42,
                ).fit(fit_weighted, labels[fit_idx])
                pred = head.predict(valid_weighted)
                predictions[(c, weight)][valid_idx] = pred
                fold_scores[(c, weight)].append(float(balanced_accuracy_score(labels[valid_idx], pred)))
        print(f"completed fold {fold}/{N_FOLDS}", flush=True)

    baseline = predictions[(1.0, 1.0)]
    if not np.array_equal(baseline, e0c["pred_model"].to_numpy()):
        raise RuntimeError("Grid's C=1, style=1 candidate did not reproduce E0c OOF predictions")
    rows = [
        {
            "C": c,
            "style_weight": weight,
            "oof_balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            **{f"fold_{i}_balanced_accuracy": fold_scores[(c, weight)][i - 1]
               for i in range(1, N_FOLDS + 1)},
        }
        for (c, weight), pred in predictions.items()
    ]
    grid = pd.DataFrame(rows).sort_values(
        ["oof_balanced_accuracy", "C", "style_weight"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_c = float(grid.loc[0, "C"])
    best_weight = float(grid.loc[0, "style_weight"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    grid.to_csv(output_dir / "grid_results.csv", index=False)
    best_oof = dev[["id", "prompt_id"]].copy()
    best_oof["true_model"] = labels
    best_oof["pred_model"] = predictions[(best_c, best_weight)]
    best_oof["fold"] = folds
    best_oof.to_csv(output_dir / "best_oof_predictions.csv", index=False)
    best_pipeline = make_pipeline()
    best_pipeline.named_steps["features"].set_params(transformer_weights={"style": best_weight})
    best_pipeline.named_steps["classifier"].set_params(C=best_c)
    best_pipeline.fit(texts, labels)
    joblib.dump(best_pipeline, output_dir / "dev_model.joblib", compress=3)
    metadata = {
        "experiment": "e0c_fixed_grid",
        "grid_predeclared": {"C": C_VALUES, "style_weight": STYLE_WEIGHTS},
        "selection_metric": "pooled OOF balanced accuracy on the development prompts",
        "selected_C": best_c,
        "selected_style_weight": best_weight,
        "selected_oof_balanced_accuracy": float(grid.loc[0, "oof_balanced_accuracy"]),
        "baseline_e0c_oof_balanced_accuracy": float(balanced_accuracy_score(labels, baseline)),
        "selected_score_is_for_model_selection": True,
        "n_prompts_holdout_untouched": int(manifest["split"].eq("holdout").sum()),
        "train_sha256": sha256(train),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(grid.to_string(index=False), flush=True)
    print(f"selected C={best_c:g}, style_weight={best_weight:g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0c-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e0c_fixed_grid"))
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.e0c_dir, args.output_dir)


if __name__ == "__main__":
    main()
