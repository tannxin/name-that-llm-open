#!/usr/bin/env python3
"""Compare the fixed family-loss RoBERTa OOF with E2d-R and E10.

Only the original 120 development prompts and their aligned five-fold OOF
scores are read. The fusion rule and its equal weights are unchanged from E10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores


def row_standardize(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if (scale <= 0).any():
        raise ValueError("A row has identical class scores")
    return centered / scale


def paired_prompt_bootstrap(
    true: np.ndarray, prompts: np.ndarray, old: np.ndarray, new: np.ndarray,
    repeats: int = 20_000, seed: int = 42,
) -> dict[str, object]:
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
        "confidence_interval_95": [float(x) for x in np.quantile(differences, [0.025, 0.975])],
        "bootstrap_unit": "prompt_id",
        "repeats": repeats,
        "seed": seed,
    }


def family(label: str) -> str:
    return "qwen" if label.startswith("qwen") else label.split("-", 1)[0]


def family_summary(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    within_family = np.asarray([family(a) == family(b) for a, b in zip(true, predicted)])
    correct = true == predicted
    return {
        "family_accuracy": float(within_family.mean()),
        "exact_given_correct_family": float(correct.sum() / within_family.sum()),
    }


def run(args: argparse.Namespace) -> None:
    original_config = json.loads(
        (args.original_roberta_scores.parent / "run_config.json").read_text(encoding="utf-8")
    )
    candidate_config = json.loads(
        (args.family_aux_scores.parent / "run_config.json").read_text(encoding="utf-8")
    )
    if candidate_config.get("family_loss_weight") != 0.2:
        raise ValueError("Expected the prespecified family loss weight 0.2")
    fixed_config_keys = (
        "model_id", "requested_revision", "epochs", "encoder_learning_rate",
        "head_learning_rate", "max_length", "batch_size", "seed", "warmup_ratio",
        "weight_decay", "dropout", "pooling", "train_sha256",
        "split_manifest_sha256", "e0_oof_sha256",
    )
    for key in fixed_config_keys:
        if original_config[key] != candidate_config[key]:
            raise ValueError(f"Original and family-loss runs differ in {key}")
    left = load_scores(args.e0c_scores)
    original = load_scores(args.original_roberta_scores)
    candidate = load_scores(args.family_aux_scores)
    if len(left) != 2160 or left["prompt_id"].nunique() != 120:
        raise ValueError("Expected the fixed 2,160-row, 120-prompt development partition")
    for other in (original, candidate):
        for key in KEY_COLUMNS:
            if not left[key].equals(other[key]):
                raise ValueError(f"OOF rows differ in {key}")
    if set(left["fold"]) != {str(value) for value in range(1, 6)}:
        raise ValueError("Expected the original five prompt-grouped folds")
    prompt_labels = left.groupby("prompt_id")["true_model"].agg(list)
    if any(len(labels) != len(LABELS) or set(labels) != set(LABELS)
           for labels in prompt_labels):
        raise ValueError("Each development prompt must have one answer per class")

    published = pd.read_csv(args.e10_oof, dtype=str, keep_default_na=False)
    for key in KEY_COLUMNS:
        if not left[key].equals(published[key]):
            raise ValueError(f"E10 OOF rows differ in {key}")
    labels = np.asarray(LABELS)
    true = left["true_model"].to_numpy()
    left_scores = left.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    original_scores = original.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    candidate_scores = candidate.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    old_direct = labels[original_scores.argmax(axis=1)]
    new_direct = labels[candidate_scores.argmax(axis=1)]
    if not np.array_equal(old_direct, original["pred_model"].to_numpy()):
        raise ValueError("Original RoBERTa OOF predictions disagree with its scores")
    if not np.array_equal(new_direct, candidate["pred_model"].to_numpy()):
        raise ValueError("Family-loss RoBERTa OOF predictions disagree with its scores")

    old_blend = labels[(row_standardize(left_scores) + row_standardize(original_scores)).argmax(axis=1)]
    if not np.array_equal(old_blend, published["zscore_pred_model"].to_numpy()):
        raise ValueError("Original score fusion does not reproduce saved E10 predictions")
    new_blend = labels[(row_standardize(left_scores) + row_standardize(candidate_scores)).argmax(axis=1)]
    prompts = left["prompt_id"].to_numpy()
    direct_comparison = paired_prompt_bootstrap(true, prompts, old_direct, new_direct)
    blend_comparison = paired_prompt_bootstrap(true, prompts, old_blend, new_blend)
    fold_rows = []
    for fold in range(1, 6):
        mask = left["fold"].eq(str(fold)).to_numpy()
        fold_rows.append({
            "fold": fold,
            "original_direct": float(balanced_accuracy_score(true[mask], old_direct[mask])),
            "family_aux_direct": float(balanced_accuracy_score(true[mask], new_direct[mask])),
            "e10_fusion": float(balanced_accuracy_score(true[mask], old_blend[mask])),
            "family_aux_fusion": float(balanced_accuracy_score(true[mask], new_blend[mask])),
        })
    old_confusion = confusion_matrix(true, old_blend, labels=LABELS)
    new_confusion = confusion_matrix(true, new_blend, labels=LABELS)
    class_metrics = pd.DataFrame({
        "model": LABELS,
        "support": old_confusion.sum(axis=1),
        "e10_recall": old_confusion.diagonal() / old_confusion.sum(axis=1),
        "family_aux_fusion_recall": new_confusion.diagonal() / new_confusion.sum(axis=1),
    })
    class_metrics["difference"] = (
        class_metrics["family_aux_fusion_recall"] - class_metrics["e10_recall"]
    )
    metrics = {
        "experiment": "e13_family_aux_fixed_weight_02",
        "family_loss_weight": 0.2,
        "original_direct_balanced_accuracy": float(balanced_accuracy_score(true, old_direct)),
        "family_aux_direct_balanced_accuracy": float(balanced_accuracy_score(true, new_direct)),
        "e10_fusion_balanced_accuracy": float(balanced_accuracy_score(true, old_blend)),
        "family_aux_fusion_balanced_accuracy": float(balanced_accuracy_score(true, new_blend)),
        "direct_comparison": direct_comparison,
        "fusion_comparison": blend_comparison,
        "original_direct_family": family_summary(true, old_direct),
        "family_aux_direct_family": family_summary(true, new_direct),
        "e10_fusion_family": family_summary(true, old_blend),
        "family_aux_fusion_family": family_summary(true, new_blend),
        "fold_scores": fold_rows,
        "n_rows_development": len(left),
        "n_prompts_development": len(prompt_labels),
        "holdout_scored": False,
        "fusion_weights_tuned": False,
        "e0c_scores_sha256": sha256(args.e0c_scores),
        "original_roberta_scores_sha256": sha256(args.original_roberta_scores),
        "family_aux_scores_sha256": sha256(args.family_aux_scores),
        "e10_oof_sha256": sha256(args.e10_oof),
        "original_run_config_sha256": sha256(args.original_roberta_scores.parent / "run_config.json"),
        "family_aux_run_config_sha256": sha256(args.family_aux_scores.parent / "run_config.json"),
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    oof = left.loc[:, list(KEY_COLUMNS)].copy()
    oof["pred_model"] = new_blend
    oof["e10_pred_model"] = old_blend
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e0c-scores", type=Path,
        default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"),
    )
    parser.add_argument(
        "--original-roberta-scores", type=Path,
        default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"),
    )
    parser.add_argument(
        "--family-aux-scores", type=Path,
        default=Path("outputs/e13_roberta_family_aux_02/oof_predictions.csv"),
    )
    parser.add_argument(
        "--e10-oof", type=Path,
        default=Path("outputs/e10_fusion_normalization/oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/e13_roberta_family_aux_02/fusion_analysis"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
