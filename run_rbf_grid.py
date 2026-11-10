#!/usr/bin/env python3
"""E7: fixed, small RBF SVC diagnostic on E0c features and saved prompt folds.

The six candidate scores use the same development folds to select a model.
They are not an unbiased estimate of the selected model's performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVC, SVC

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import load_e0_split


C_VALUES = (1.0, 10.0)
GAMMA_MULTIPLIERS = (0.5, 1.0, 2.0)
DISTANCE_SAMPLE_SIZE = 512
DISTANCE_SAMPLE_SEED = 4200


def median_training_distance(matrix, seed: int) -> tuple[float, int]:
    """Estimate median nonself squared distance from train rows only."""
    rng = np.random.default_rng(seed)
    n_sample = min(DISTANCE_SAMPLE_SIZE, matrix.shape[0])
    sample_idx = np.sort(rng.choice(matrix.shape[0], n_sample, replace=False))
    sample = matrix[sample_idx]
    distances = euclidean_distances(sample, sample, squared=True)
    upper = distances[np.triu_indices(n_sample, k=1)]
    median = float(np.median(upper))
    if not np.isfinite(median) or median <= 0:
        raise ValueError(f"Invalid train-fold median squared distance: {median}")
    return median, n_sample


def run(train: Path, e0_dir: Path, e0c_dir: Path, output_dir: Path) -> None:
    frame = load_and_validate(train)
    dev, manifest, e0, folds, _ = load_e0_split(frame, e0_dir, train)
    if output_dir.resolve() in {e0_dir.resolve(), e0c_dir.resolve()}:
        raise ValueError("E7 output directory must differ from baseline directories")
    e0c = pd.read_csv(e0c_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for column in ("id", "prompt_id", "true_model", "fold"):
        if column not in e0c or not e0c[column].equals(e0[column]):
            raise ValueError(f"E0c {column} does not match the saved E0 development split")
    if "pred_model" not in e0c:
        raise ValueError("E0c OOF predictions are missing")
    e0c_metrics = json.loads((e0c_dir / "metrics.json").read_text(encoding="utf-8"))
    if e0c_metrics["train_sha256"] != sha256(train):
        raise ValueError("E0c and input training file hashes differ")

    labels = dev["model"].to_numpy()
    texts = dev["text"].to_numpy()
    candidates = [(c, multiplier) for c in C_VALUES for multiplier in GAMMA_MULTIPLIERS]
    predictions = {key: np.empty(len(dev), dtype=object) for key in candidates}
    fold_scores = {key: [] for key in candidates}
    fold_details: list[dict] = []
    linear_reproduction = np.empty(len(dev), dtype=object)

    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} does not contain every official label")
        fit_prompts = set(dev.loc[fit_idx, "prompt_id"])
        valid_prompts = set(dev.loc[valid_idx, "prompt_id"])
        if fit_prompts & valid_prompts:
            raise ValueError(f"Fold {fold} has overlapping prompt groups")

        # The exact E0c feature pipeline is fitted on this fold's training rows.
        features = make_pipeline().named_steps["features"]
        x_fit = features.fit_transform(texts[fit_idx])
        x_valid = features.transform(texts[valid_idx])
        linear = LinearSVC(
            C=1.0, class_weight="balanced", dual=True,
            max_iter=10_000, random_state=42,
        ).fit(x_fit, labels[fit_idx])
        linear_reproduction[valid_idx] = linear.predict(x_valid)

        median_distance, n_sample = median_training_distance(
            x_fit, DISTANCE_SAMPLE_SEED + fold,
        )
        # Cache the fold's pairwise squared distances once. LibSVM can then
        # reuse each RBF Gram matrix for both C values instead of recomputing
        # sparse high-dimensional distances for all six fits.
        train_distances = euclidean_distances(x_fit, x_fit, squared=True)
        valid_distances = euclidean_distances(x_valid, x_fit, squared=True)
        fold_details.append({
            "fold": fold,
            "train_rows": len(fit_idx),
            "validation_rows": len(valid_idx),
            "train_prompts": len(fit_prompts),
            "validation_prompts": len(valid_prompts),
            "distance_sample_rows": n_sample,
            "median_train_squared_distance": median_distance,
        })
        for multiplier in GAMMA_MULTIPLIERS:
            # gamma has units of inverse squared distance. The multiplier grid
            # is fixed before looking at any validation labels or scores.
            gamma = multiplier / median_distance
            train_kernel = np.exp(-gamma * train_distances)
            valid_kernel = np.exp(-gamma * valid_distances)
            for c in C_VALUES:
                model = SVC(
                    C=c, kernel="precomputed", class_weight="balanced",
                    cache_size=1024, random_state=42,
                ).fit(train_kernel, labels[fit_idx])
                pred = model.predict(valid_kernel)
                predictions[(c, multiplier)][valid_idx] = pred
                fold_scores[(c, multiplier)].append(
                    float(balanced_accuracy_score(labels[valid_idx], pred))
                )
        print(f"E7 completed fold {fold}/{N_FOLDS}", flush=True)

    if not np.array_equal(linear_reproduction, e0c["pred_model"].to_numpy()):
        mismatch = int(np.sum(linear_reproduction != e0c["pred_model"].to_numpy()))
        raise RuntimeError(f"E0c pipeline reproduction differs in {mismatch} OOF rows")
    baseline_score = float(balanced_accuracy_score(labels, linear_reproduction))
    if not np.isclose(baseline_score, e0c_metrics["oof_balanced_accuracy"]):
        raise RuntimeError("Reproduced E0c score differs from saved metrics")
    rows = [
        {
            "C": c,
            "gamma_multiplier": multiplier,
            "oof_balanced_accuracy": float(balanced_accuracy_score(labels, predictions[(c, multiplier)])),
            **{
                f"fold_{i}_balanced_accuracy": fold_scores[(c, multiplier)][i - 1]
                for i in range(1, N_FOLDS + 1)
            },
        }
        for c, multiplier in candidates
    ]
    grid = pd.DataFrame(rows).sort_values(
        ["oof_balanced_accuracy", "C", "gamma_multiplier"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_c = float(grid.loc[0, "C"])
    best_multiplier = float(grid.loc[0, "gamma_multiplier"])
    best_pred = predictions[(best_c, best_multiplier)]
    if any(value is None for value in best_pred):
        raise RuntimeError("Best OOF predictions are incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    grid.to_csv(output_dir / "grid_results.csv", index=False)
    pd.DataFrame(fold_details).to_csv(output_dir / "fold_details.csv", index=False)
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["pred_model"] = best_pred
    oof["fold"] = folds
    oof["correct"] = labels == best_pred
    oof.to_csv(output_dir / "best_oof_predictions.csv", index=False)

    # Refit the selected model on the full development partition, still without
    # accessing any holdout response text during feature fitting or training.
    final_features = make_pipeline().named_steps["features"]
    x_dev = final_features.fit_transform(texts)
    final_median, final_n_sample = median_training_distance(
        x_dev, DISTANCE_SAMPLE_SEED,
    )
    final_gamma = best_multiplier / final_median
    final_head = SVC(
        C=best_c, gamma=final_gamma, kernel="rbf", class_weight="balanced",
        cache_size=1024, random_state=42,
    ).fit(x_dev, labels)
    joblib.dump((final_features, final_head), output_dir / "dev_model.joblib", compress=3)
    metrics = {
        "experiment": "e7_rbf_svc",
        "metric": "balanced_accuracy",
        "grid_predeclared": {
            "C": C_VALUES,
            "gamma_multiplier_over_median_train_squared_distance": GAMMA_MULTIPLIERS,
            "style_block_weight": 1.0,
            "distance_sample_size": DISTANCE_SAMPLE_SIZE,
            "distance_sample_seed": DISTANCE_SAMPLE_SEED,
        },
        "selected_C": best_c,
        "selected_gamma_multiplier": best_multiplier,
        "selected_oof_balanced_accuracy": float(grid.loc[0, "oof_balanced_accuracy"]),
        "baseline_e0c_oof_balanced_accuracy": baseline_score,
        "baseline_predictions_exactly_reproduced": True,
        "selection_score_is_biased": True,
        "selection_note": "Six configurations were selected on these same five development folds. Evaluate the locked 30-prompt holdout once after the broader model choice is frozen.",
        "final_development_fit_gamma": final_gamma,
        "final_development_fit_median_squared_distance": final_median,
        "final_development_distance_sample_rows": final_n_sample,
        "n_rows_development": len(dev),
        "n_prompts_development": int(dev["prompt_id"].nunique()),
        "n_prompts_holdout_untouched": int(manifest["split"].eq("holdout").sum()),
        "train_sha256": sha256(train),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(grid.to_string(index=False), flush=True)
    print(f"E7 selected C={best_c:g}, gamma multiplier={best_multiplier:g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0c-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e7_rbf_svc"))
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.e0c_dir, args.output_dir)


if __name__ == "__main__":
    main()
