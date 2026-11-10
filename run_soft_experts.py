#!/usr/bin/env python3
"""E11: fixed two-pair soft experts on the original prompt-grouped OOF split.

The 30-prompt reserved holdout is never fit or scored. Pair choices, gate,
normalization, and weight are fixed in docs/e11_soft_experts.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores, ranks
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


PAIRS = (
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("glm-5.3", "glm-5.3-flash"),
)
EXPERT_WEIGHT = 0.5
GATE_DISTANCE = 2.0
SCENARIOS = ("gemini", "glm", "both")


def load_input(
    train_path: Path, e0_dir: Path, left_path: Path, right_path: Path,
    rank_path: Path, zscore_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = load_and_validate(train_path)
    development, _, baseline, folds, _ = load_e0_split(frame, e0_dir, train_path)
    left, right = load_scores(left_path), load_scores(right_path)
    rank = pd.read_csv(rank_path, dtype=str, keep_default_na=False)
    zscore = pd.read_csv(zscore_path, dtype=str, keep_default_na=False)
    if len(development) != 2160 or development["prompt_id"].nunique() != 120:
        raise ValueError("Expected the original 2,160 development rows and 120 prompts")
    if len(left) != len(development):
        raise ValueError("OOF score length differs from the fixed development split")
    for key in KEY_COLUMNS:
        if not baseline[key].equals(left[key]) or not left[key].equals(right[key]):
            raise ValueError(f"The saved OOF score files differ in {key}")
        if not left[key].equals(rank[key]) or not left[key].equals(zscore[key]):
            raise ValueError(f"The saved fusion references differ in {key}")
    if not np.array_equal(folds.astype(str), left["fold"].to_numpy()):
        raise ValueError("The saved OOF fold assignment changed")
    if set(left["true_model"]) != set(LABELS):
        raise ValueError("The OOF label set changed")
    per_prompt = left.groupby("prompt_id")["true_model"].agg(list)
    if any(len(values) != len(LABELS) or set(values) != set(LABELS) for values in per_prompt):
        raise ValueError("Each development prompt must contain exactly one of each class")
    left_scores = left.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    right_scores = right.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    left_rank = ranks(left_scores)
    rank_base = left_rank + ranks(right_scores) + left_rank * 1e-4
    zscore_base = row_standardize(left_scores) + row_standardize(right_scores)
    label_array = np.asarray(LABELS)
    if not np.array_equal(label_array[rank_base.argmax(axis=1)], rank["pred_model"].to_numpy()):
        raise ValueError("Rank reference did not exactly reproduce E7e-R")
    if not np.array_equal(label_array[zscore_base.argmax(axis=1)], zscore["zscore_pred_model"].to_numpy()):
        raise ValueError("Z-score reference did not exactly reproduce E10")
    return development, folds, left["true_model"].to_numpy(), rank_base, zscore_base


def fit_expert_margins(
    development: pd.DataFrame, folds: np.ndarray, labels: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    texts = development["text"].to_numpy()
    margins = np.full((len(development), len(PAIRS)), np.nan, dtype=np.float64)
    training_metadata = []
    for fold in range(1, N_FOLDS + 1):
        fit_fold = folds != fold
        validation = folds == fold
        if set(development.loc[fit_fold, "prompt_id"]) & set(development.loc[validation, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompt groups")
        for pair_index, (first, second) in enumerate(PAIRS):
            fit_pair = fit_fold & np.isin(labels, (first, second))
            if set(labels[fit_pair]) != {first, second}:
                raise ValueError(f"Fold {fold} is missing a pair class")
            model = make_pipeline()
            model.fit(texts[fit_pair], labels[fit_pair])
            classes = tuple(model.named_steps["classifier"].classes_)
            if set(classes) != {first, second}:
                raise ValueError(f"Fold {fold} expert learned unexpected classes")
            sign_for_first = 1 if classes[1] == first else -1
            fit_margin = sign_for_first * np.asarray(model.decision_function(texts[fit_pair]))
            scale = float(np.median(np.abs(fit_margin)))
            if scale <= 0 or not np.isfinite(scale):
                scale = 1.0
            validation_margin = sign_for_first * np.asarray(model.decision_function(texts[validation]))
            margins[validation, pair_index] = np.tanh(validation_margin / scale)
            training_metadata.append({
                "fold": fold,
                "pair": [first, second],
                "training_rows": int(fit_pair.sum()),
                "validation_rows": int(validation.sum()),
                "training_prompts": int(development.loc[fit_pair, "prompt_id"].nunique()),
                "validation_prompts": int(development.loc[validation, "prompt_id"].nunique()),
                "training_median_absolute_margin": scale,
            })
            print(f"fold={fold} pair={first}/{second} fitted", flush=True)
    if not np.isfinite(margins).all():
        raise RuntimeError("One or more OOF expert margins are missing")
    return margins, training_metadata


def adjust(base: np.ndarray, margins: np.ndarray, selected: tuple[int, ...]) -> np.ndarray:
    adjusted = base.copy()
    top = base.max(axis=1)
    for pair_index in selected:
        first, second = PAIRS[pair_index]
        first_col, second_col = LABELS.index(first), LABELS.index(second)
        best_pair = np.maximum(base[:, first_col], base[:, second_col])
        gate = np.exp(-np.maximum(0.0, top - best_pair) / GATE_DISTANCE)
        delta = EXPERT_WEIGHT * gate * margins[:, pair_index]
        adjusted[:, first_col] += delta
        adjusted[:, second_col] -= delta
    return adjusted


def run(
    train_path: Path, e0_dir: Path, left_path: Path, right_path: Path,
    rank_path: Path, zscore_path: Path, output_dir: Path,
) -> None:
    development, folds, labels, rank_base, zscore_base = load_input(
        train_path, e0_dir, left_path, right_path, rank_path, zscore_path
    )
    margins, training_metadata = fit_expert_margins(development, folds, labels)
    label_array = np.asarray(LABELS)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = development[["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["fold"] = folds
    oof["gemini_expert_margin"] = margins[:, 0]
    oof["glm_expert_margin"] = margins[:, 1]
    fold_metrics = []
    class_metrics = []
    scenario_metrics = []
    variant_pairs = {
        "none": (), "gemini": (0,), "glm": (1,), "both": (0, 1),
    }
    prompts = development["prompt_id"].to_numpy()
    for fusion, base in (("rank", rank_base), ("zscore", zscore_base)):
        reference = label_array[base.argmax(axis=1)]
        reference_score = float(balanced_accuracy_score(labels, reference))
        for scenario, selected in variant_pairs.items():
            predicted = label_array[adjust(base, margins, selected).argmax(axis=1)]
            oof[f"pred_{fusion}_{scenario}"] = predicted
            confusion = confusion_matrix(labels, predicted, labels=LABELS)
            class_recall = confusion.diagonal() / confusion.sum(axis=1)
            score = float(balanced_accuracy_score(labels, predicted))
            class_metrics.extend({
                "fusion": fusion,
                "scenario": scenario,
                "model": label,
                "support": int(confusion.sum(axis=1)[index]),
                "recall": float(class_recall[index]),
            } for index, label in enumerate(LABELS))
            for fold in range(1, N_FOLDS + 1):
                mask = folds == fold
                fold_metrics.append({
                    "fusion": fusion,
                    "scenario": scenario,
                    "fold": fold,
                    "balanced_accuracy": float(balanced_accuracy_score(labels[mask], predicted[mask])),
                    "difference_vs_no_expert": float(
                        balanced_accuracy_score(labels[mask], predicted[mask])
                        - balanced_accuracy_score(labels[mask], reference[mask])
                    ),
                })
            scenario_metrics.append({
                "fusion": fusion,
                "scenario": scenario,
                "pooled_balanced_accuracy": score,
                "difference_vs_no_expert": score - reference_score,
                "paired_prompt_bootstrap_vs_no_expert": paired_prompt_bootstrap(
                    labels, prompts, reference, predicted
                ) if scenario != "none" else None,
            })
            pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
                index="true_model", columns="pred_model"
            ).to_csv(output_dir / f"confusion_{fusion}_{scenario}.csv")
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(class_metrics).to_csv(output_dir / "class_metrics.csv", index=False)
    metadata = {
        "experiment": "e11_fixed_two_pair_soft_experts",
        "predeclared_protocol": "docs/e11_soft_experts.md",
        "pairs": [list(pair) for pair in PAIRS],
        "expert_weight": EXPERT_WEIGHT,
        "gate_distance": GATE_DISTANCE,
        "margin_scaling": "median absolute training-fold margin, then tanh",
        "feature_classifier": "original E0c features + binary balanced LinearSVC",
        "training_metadata": training_metadata,
        "scenario_metrics": scenario_metrics,
        "n_rows_development": len(development),
        "n_prompts_development": int(development["prompt_id"].nunique()),
        "n_folds": N_FOLDS,
        "rank_reference_reproduced": True,
        "zscore_reference_reproduced": True,
        "holdout_scored": False,
        "train_sha256": sha256(train_path),
        "e0_score_sha256": sha256(left_path),
        "roberta_score_sha256": sha256(right_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scenario_metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--left", type=Path, default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    parser.add_argument("--right", type=Path, default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    parser.add_argument("--rank-reference", type=Path, default=Path("outputs/e7_blend_e0c_roberta_ft_scalerfix/oof_predictions.csv"))
    parser.add_argument("--zscore-reference", type=Path, default=Path("outputs/e10_fusion_normalization/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e11_soft_experts"))
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.left, args.right, args.rank_reference, args.zscore_reference, args.output_dir)


if __name__ == "__main__":
    main()
