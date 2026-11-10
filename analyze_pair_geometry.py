#!/usr/bin/env python3
"""Describe development-only model-pair geometry and E10 OOF confusions.

This is a diagnostic, not a new classifier. The fixed prompt holdout is excluded
before vectorization, scaling, distance computation, and reporting. Distances
are descriptive and must be recomputed inside training folds if used later to
set a loss weight or to select a model.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from run_e0 import LABELS, load_and_validate, sha256
from run_style import FEATURE_NAMES, extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/e0_tfidf_svm/split_manifest.csv")
    )
    parser.add_argument(
        "--e0-metrics", type=Path, default=Path("outputs/e0_tfidf_svm/metrics.json")
    )
    parser.add_argument(
        "--e10-oof", type=Path,
        default=Path("outputs/e10_fusion_normalization/oof_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pair_geometry_diagnostic"))
    parser.add_argument("--report", type=Path, default=Path("docs/pair_geometry_diagnostic.md"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def load_development(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    e0_metrics = json.loads(args.e0_metrics.read_text(encoding="utf-8"))
    if e0_metrics["train_sha256"] != sha256(args.train):
        raise ValueError("Training CSV differs from the fixed E0 split")
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("Unexpected split manifest columns")
    if manifest["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt in manifest")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("The manifest must have development and holdout groups")
    frame = load_and_validate(args.train)
    if set(frame["prompt_id"]) != set(manifest["prompt_id"]):
        raise ValueError("Manifest does not cover all training prompts")
    dev_prompts = set(manifest.loc[manifest["split"].eq("development"), "prompt_id"])
    dev = frame.loc[frame["prompt_id"].isin(dev_prompts)].reset_index(drop=True)
    if dev["prompt_id"].nunique() != 120 or len(dev) != 2160:
        raise ValueError("Expected the fixed 120-prompt, 2160-row development split")
    table = dev.groupby("prompt_id")["model"].agg(["size", "nunique"])
    if (table["size"] != len(LABELS)).any() or (table["nunique"] != len(LABELS)).any():
        raise ValueError("Each development prompt must contain all 18 labels once")

    oof = pd.read_csv(args.e10_oof, dtype=str, keep_default_na=False)
    required = {"id", "prompt_id", "true_model", "fold", "zscore_pred_model"}
    if not required.issubset(oof.columns) or oof["id"].duplicated().any():
        raise ValueError("Unexpected E10 OOF schema or duplicate id")
    if len(oof) != len(dev) or set(oof["id"]) != set(dev["id"]):
        raise ValueError("E10 OOF IDs do not exactly match development IDs")
    aligned = dev.loc[:, ["id", "prompt_id", "model"]].merge(
        oof.loc[:, ["id", "prompt_id", "true_model", "fold", "zscore_pred_model"]],
        on="id", how="left", validate="one_to_one", suffixes=("", "_oof"),
    )
    if not aligned["prompt_id"].eq(aligned["prompt_id_oof"]).all():
        raise ValueError("E10 OOF prompt IDs differ from development")
    if not aligned["model"].eq(aligned["true_model"]).all():
        raise ValueError("E10 OOF labels differ from development")
    if not set(aligned["zscore_pred_model"]).issubset(set(LABELS)):
        raise ValueError("E10 OOF contains an unknown prediction label")
    folds = pd.to_numeric(aligned["fold"], errors="raise").to_numpy()
    if set(folds) != set(range(1, 6)):
        raise ValueError("E10 OOF must contain all five folds")
    if (aligned.assign(fold=folds).groupby("prompt_id")["fold"].nunique() != 1).any():
        raise ValueError("A prompt appears in more than one E10 fold")
    score = balanced_accuracy_score(aligned["model"], aligned["zscore_pred_model"])
    if not np.isclose(score, 0.399537037037037):
        raise ValueError(f"Unexpected E10 OOF score: {score}")
    return dev, aligned


def consistency_ratio(deltas) -> float:
    """Norm of the mean paired difference / RMS norm of paired differences."""
    if hasattr(deltas, "multiply"):
        mean = np.asarray(deltas.mean(axis=0)).ravel()
        mean_sq_norm = float(np.asarray(deltas.multiply(deltas).sum(axis=1)).mean())
    else:
        mean = deltas.mean(axis=0)
        mean_sq_norm = float(np.mean(np.sum(deltas * deltas, axis=1)))
    return float(np.linalg.norm(mean) / np.sqrt(mean_sq_norm)) if mean_sq_norm > 0 else 0.0


def interval(values: np.ndarray, samples: np.ndarray) -> tuple[float, float]:
    means = values[samples].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def markdown_table(frame: pd.DataFrame, columns: list[str], n: int) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in frame.head(n).itertuples(index=False):
        values = row._asdict()
        lines.append("| " + " | ".join(str(values[name]) for name in columns) + " |")
    return lines


def run(args: argparse.Namespace) -> None:
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 prompt bootstrap draws")
    dev, aligned = load_development(args)
    prompts = sorted(dev["prompt_id"].unique())
    n_prompts = len(prompts)
    index = dev.reset_index().pivot(index="prompt_id", columns="model", values="index")
    index = index.loc[prompts, list(LABELS)].to_numpy(dtype=int)

    # All preprocessing inputs here are development rows; no holdout text is
    # vectorized, scaled, scored, or included in any output.
    char_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100_000,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )
    chars = char_vectorizer.fit_transform(dev["text"])
    style_raw = extract_features(dev["text"])
    style = StandardScaler().fit_transform(style_raw)
    n_style = len(FEATURE_NAMES)

    pairs = list(combinations(range(len(LABELS)), 2))
    char_distance = np.zeros((n_prompts, len(pairs)), dtype=np.float64)
    style_distance = np.zeros_like(char_distance)
    for p, row in enumerate(index):
        char_sim = (chars[row] @ chars[row].T).toarray()
        char_sim = np.clip(char_sim, 0, 1)
        char_norm = np.maximum(np.diag(char_sim), 1e-12)
        char_sim /= np.sqrt(np.outer(char_norm, char_norm))
        style_diff = style[row, None, :] - style[None, row, :]
        style_dist = np.linalg.norm(style_diff, axis=2) / np.sqrt(n_style)
        for k, (i, j) in enumerate(pairs):
            char_distance[p, k] = 1 - char_sim[i, j]
            style_distance[p, k] = style_dist[i, j]

    char_percentile = np.array([rankdata(row, method="average") for row in char_distance])
    style_percentile = np.array([rankdata(row, method="average") for row in style_distance])
    char_percentile = (char_percentile - 1) / (len(pairs) - 1)
    style_percentile = (style_percentile - 1) / (len(pairs) - 1)

    confusion = confusion_matrix(
        aligned["model"], aligned["zscore_pred_model"], labels=LABELS,
    )
    fold_by_prompt = aligned.groupby("prompt_id")["fold"].first().loc[prompts].astype(int).to_numpy()
    rng = np.random.default_rng(42)
    bootstrap_samples = rng.integers(0, n_prompts, size=(args.bootstrap, n_prompts))
    records = []
    for k, (i, j) in enumerate(pairs):
        a, b = LABELS[i], LABELS[j]
        char_delta = chars[index[:, i]] - chars[index[:, j]]
        style_delta = style[index[:, i]] - style[index[:, j]]
        a_to_b = int(confusion[i, j])
        b_to_a = int(confusion[j, i])
        per_prompt_confusion = (
            (aligned.loc[aligned["model"].eq(a)].set_index("prompt_id")
             .loc[prompts, "zscore_pred_model"].to_numpy() == b).astype(float)
            + (aligned.loc[aligned["model"].eq(b)].set_index("prompt_id")
               .loc[prompts, "zscore_pred_model"].to_numpy() == a).astype(float)
        ) / 2
        char_ci = interval(char_percentile[:, k], bootstrap_samples)
        style_ci = interval(style_percentile[:, k], bootstrap_samples)
        confusion_ci = interval(per_prompt_confusion, bootstrap_samples)
        char_folds = [float(char_percentile[fold_by_prompt == f, k].mean()) for f in range(1, 6)]
        style_folds = [float(style_percentile[fold_by_prompt == f, k].mean()) for f in range(1, 6)]
        records.append({
            "model_a": a, "model_b": b,
            "char_cosine_distance_mean": float(char_distance[:, k].mean()),
            "char_closeness_percentile_mean": float(char_percentile[:, k].mean()),
            "char_closeness_ci_low": char_ci[0], "char_closeness_ci_high": char_ci[1],
            "char_closeness_fold_min": min(char_folds),
            "char_closeness_fold_max": max(char_folds),
            "char_paired_direction_consistency": consistency_ratio(char_delta),
            "style_scaled_euclidean_mean": float(style_distance[:, k].mean()),
            "style_closeness_percentile_mean": float(style_percentile[:, k].mean()),
            "style_closeness_ci_low": style_ci[0], "style_closeness_ci_high": style_ci[1],
            "style_closeness_fold_min": min(style_folds),
            "style_closeness_fold_max": max(style_folds),
            "style_paired_direction_consistency": consistency_ratio(style_delta),
            "oof_a_to_b": a_to_b, "oof_b_to_a": b_to_a,
            "oof_mutual_confusion_count": a_to_b + b_to_a,
            "oof_mutual_confusion_rate": float(per_prompt_confusion.mean()),
            "oof_mutual_confusion_ci_low": confusion_ci[0],
            "oof_mutual_confusion_ci_high": confusion_ci[1],
        })
    pair_frame = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).to_csv(
        args.output_dir / "e10_oof_confusion.csv", index_label="true_model"
    )
    for name, distances in (("char", char_distance), ("style", style_distance)):
        matrix = np.zeros((len(LABELS), len(LABELS)))
        for k, (i, j) in enumerate(pairs):
            matrix[i, j] = matrix[j, i] = float(distances[:, k].mean())
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
            args.output_dir / f"{name}_distance_matrix.csv", index_label="model"
        )

    char_rho = spearmanr(pair_frame["char_closeness_percentile_mean"],
                         pair_frame["oof_mutual_confusion_rate"]).statistic
    style_rho = spearmanr(pair_frame["style_closeness_percentile_mean"],
                          pair_frame["oof_mutual_confusion_rate"]).statistic
    within_family = (
        pair_frame["model_a"].str.split("-").str[0]
        == pair_frame["model_b"].str.split("-").str[0]
    )
    within = pair_frame.loc[within_family]
    cross = pair_frame.loc[~within_family]
    within_char_rho = spearmanr(
        within["char_closeness_percentile_mean"], within["oof_mutual_confusion_rate"]
    ).statistic
    within_style_rho = spearmanr(
        within["style_closeness_percentile_mean"], within["oof_mutual_confusion_rate"]
    ).statistic
    cross_char_rho = spearmanr(
        cross["char_closeness_percentile_mean"], cross["oof_mutual_confusion_rate"]
    ).statistic
    cross_style_rho = spearmanr(
        cross["style_closeness_percentile_mean"], cross["oof_mutual_confusion_rate"]
    ).statistic
    summary = {
        "train_sha256": sha256(args.train), "development_prompts": n_prompts,
        "development_rows": len(dev), "holdout_prompts_processed": 0,
        "e10_oof_balanced_accuracy": float(balanced_accuracy_score(
            aligned["model"], aligned["zscore_pred_model"])),
        "char_vocabulary_size": len(char_vectorizer.vocabulary_),
        "style_feature_count": n_style,
        "pair_count": len(pairs),
        "spearman_closeness_vs_mutual_confusion_char": float(char_rho),
        "spearman_closeness_vs_mutual_confusion_style": float(style_rho),
        "within_family_pair_count": len(within),
        "spearman_within_family_char": float(within_char_rho),
        "spearman_within_family_style": float(within_style_rho),
        "spearman_cross_family_char": float(cross_char_rho),
        "spearman_cross_family_style": float(cross_style_rho),
        "random_sign_direction_consistency_reference": float(1 / np.sqrt(n_prompts)),
        "bootstrap_draws": args.bootstrap,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    top = pair_frame.sort_values("oof_mutual_confusion_count", ascending=False).head(12).copy()
    top["pair"] = top["model_a"] + " ↔ " + top["model_b"]
    top["errors"] = top["oof_a_to_b"].astype(str) + "+" + top["oof_b_to_a"].astype(str)
    top["char_pct"] = (100 * top["char_closeness_percentile_mean"]).round(1).astype(str)
    top["style_pct"] = (100 * top["style_closeness_percentile_mean"]).round(1).astype(str)
    top["char_R"] = top["char_paired_direction_consistency"].round(3).astype(str)
    top["style_R"] = top["style_paired_direction_consistency"].round(3).astype(str)
    lines = [
        "# Development-only pair geometry diagnostic", "",
        "This report uses the fixed 120 development prompts (2,160 answers, 120 per class) "
        "and the saved E10 grouped out-of-fold predictions. The 30 held-out prompts were "
        "excluded before feature extraction, vectorizer fitting, scaling, and distance computation. "
        "No new classifier was trained and no Kaggle test labels were used.", "",
        "## Measures", "",
        "- Character distance is `1 − cosine` for character 3–5-gram TF-IDF. The 100,000-feature "
        "vocabulary is fitted on development text only. Style distance is Euclidean distance "
        "over the existing 57 standardized style features, divided by √57.",
        "- Closeness percentile ranks each pair's distance among all 153 model pairs *within the same prompt*. "
        "0 is the closest pair and 100 the most distant. Values below 50 mean a pair is more "
        "similar than a typical pair in the same-prompt comparison. The table reports means "
        "across prompts; 95% prompt-bootstrap intervals and five-fold ranges are in the CSV.",
        "- Direction consistency `R = ||mean(x_a − x_b)|| / sqrt(mean(||x_a − x_b||²))` ranges from "
        "0 to 1. Higher values mean the pair's feature difference points in a more stable "
        f"direction across prompts. With 120 prompts, random sign flips have R² expectation "
        f"1/120, corresponding to an RMS reference of {1 / np.sqrt(n_prompts):.3f} for R. "
        "Values near this reference offer little evidence of a stable direction. R is a "
        "descriptive signal, not a classification score.",
        "- `errors` gives E10 OOF predictions A→B + B→A. The two directions matter; proximity "
        "alone does not establish that increasing a loss weight will help.", "",
        "## Most confused pairs", "",
    ]
    lines += markdown_table(top, ["pair", "errors", "char_pct", "style_pct", "char_R", "style_R"], 12)
    lines += ["", "## Overall relationship", "",
              f"Across all 153 pairs, Spearman correlation between closeness percentile "
              f"and mutual OOF confusion rate is **{char_rho:.3f}** for character TF-IDF "
              f"and **{style_rho:.3f}** for style features. Negative values mean closer pairs "
              "tend to be confused more often. This is descriptive because the same development "
              "set informed the geometry and earlier model choices.", "",
              f"Within the {len(within)} same-family pairs, the correlations are "
              f"**{within_char_rho:.3f}** (character) and **{within_style_rho:.3f}** "
              f"(style). Across the {len(cross)} cross-family pairs they are "
              f"**{cross_char_rho:.3f}** and **{cross_style_rho:.3f}**, respectively. "
              "Thus the overall relationship is not solely a family-versus-non-family effect.", "",
              "The full 18×18 mean distance matrices, 18×18 E10 confusion matrix, and all 153 "
              "pair metrics (including prompt-bootstrap intervals and fold ranges) are in "
              "`outputs/pair_geometry_diagnostic/` (Git-ignored).", "",
              "## Interpretation for later experiments", "",
              "Use the same-prompt distances to identify candidates, then compare them with "
              "directed OOF confusion and direction consistency. A very similar pair can still "
              "be separable if its differences are stable; a distant pair can remain hard if its "
              "difference changes with prompt. Any proposed loss weights must be estimated "
              "inside each training fold without validation labels, fixed before evaluating that "
              "fold, and checked against unweighted cross-entropy under the same backbone and "
              "training schedule. This diagnostic is not an unbiased estimate of a new loss's gain.", "",
              "The competition test file has only `id,text`; same-prompt comparison is available "
              "in training, but cannot be required at inference.", "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.report} and {args.output_dir}")


if __name__ == "__main__":
    run(parse_args())
