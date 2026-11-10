#!/usr/bin/env python3
"""E23: one preregistered, unlabeled-batch marginal correction of E16 OOF.

The correction function accepts only an unlabeled score matrix. True labels
and prompt IDs are used exclusively to verify the OOF protocol and score it.
Protocol: docs/e23_marginal_correction.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS, N_FOLDS
from run_e22_layout_evidence import read_inputs
from run_score_normalization import paired_prompt_bootstrap


OUTPUT_DIR = Path("outputs/e23_marginal_correction")
LOGIT_STRENGTH = 0.10
LOG_RATIO_LIMIT = 1.0


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def correct_unlabeled_batch(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return corrected logits, initial soft marginal, and one fixed bias.

    No labels, IDs, true class frequencies, or prompt grouping enter here.
    """
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or logits.shape[0] < 1:
        raise ValueError("Expected a nonempty batch with exactly 18 class logits")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite input logit")
    marginal = softmax(logits).mean(axis=0)
    uniform = 1.0 / len(LABELS)
    log_ratio = np.log(marginal / uniform)
    bias = -LOGIT_STRENGTH * np.clip(log_ratio, -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
    if np.max(np.abs(bias)) > LOGIT_STRENGTH + 1e-12:
        raise AssertionError("Preregistered logit-shift cap failed")
    return logits + bias[None, :], marginal, bias


def run(output_dir: Path) -> dict:
    dev, folds, true, fused, hashes = read_inputs()
    logits = fused / 3.0
    base_pred = np.asarray(LABELS)[np.argmax(logits, axis=1)]
    candidate = np.empty_like(logits)
    marginal_rows: list[dict] = []
    fold_rows: list[dict] = []

    for fold in range(1, N_FOLDS + 1):
        train = folds != fold
        valid = folds == fold
        if int(train.sum()) != 1728 or int(valid.sum()) != 432:
            raise ValueError(f"Fold {fold} row counts changed")
        if dev.loc[train, "prompt_id"].nunique() != 96 or dev.loc[valid, "prompt_id"].nunique() != 24:
            raise ValueError(f"Fold {fold} prompt counts changed")
        if set(dev.loc[train, "prompt_id"]) & set(dev.loc[valid, "prompt_id"]):
            raise ValueError(f"Fold {fold} train and validation prompts overlap")

        # The only function computing a correction takes the 432 score rows.
        # Neither true[valid] nor dev.loc[valid, 'prompt_id'] is passed to it.
        adjusted, marginal, bias = correct_unlabeled_batch(logits[valid])
        candidate[valid] = adjusted
        old = base_pred[valid]
        new = np.asarray(LABELS)[adjusted.argmax(axis=1)]
        fold_rows.append({
            "fold": fold,
            "validation_rows": int(valid.sum()),
            "validation_prompts": 24,
            "e16_balanced_accuracy": float(balanced_accuracy_score(true[valid], old)),
            "e23_balanced_accuracy": float(balanced_accuracy_score(true[valid], new)),
            "changed_predictions": int((old != new).sum()),
            "mean_abs_logit_shift": float(np.abs(bias).mean()),
            "max_abs_logit_shift": float(np.abs(bias).max()),
            "pairwise_margin_shift_range": float(np.ptp(bias)),
        })
        for j, label in enumerate(LABELS):
            before = int((old == label).sum())
            after = int((new == label).sum())
            marginal_rows.append({
                "fold": fold,
                "model": label,
                "unlabeled_soft_share_before": float(marginal[j]),
                "hard_count_before": before,
                "hard_share_before": before / int(valid.sum()),
                "hard_count_after": after,
                "hard_share_after": after / int(valid.sum()),
                "log_ratio_to_uniform": float(np.log(marginal[j] * len(LABELS))),
                "logit_shift": float(bias[j]),
            })

    new_pred = np.asarray(LABELS)[candidate.argmax(axis=1)]
    fold_table = pd.DataFrame(fold_rows)
    fold_table["delta"] = fold_table["e23_balanced_accuracy"] - fold_table["e16_balanced_accuracy"]
    class_rows = []
    for label in LABELS:
        mask = true == label
        old_recall = float((base_pred[mask] == label).mean())
        new_recall = float((new_pred[mask] == label).mean())
        class_rows.append({
            "model": label,
            "support": int(mask.sum()),
            "e16_recall": old_recall,
            "e23_recall": new_recall,
            "delta": new_recall - old_recall,
        })
    class_table = pd.DataFrame(class_rows)
    marginal_table = pd.DataFrame(marginal_rows)
    old_ba = float(balanced_accuracy_score(true, base_pred))
    new_ba = float(balanced_accuracy_score(true, new_pred))
    if abs(old_ba - 0.40925925925925927) > 1e-12:
        raise AssertionError("E16 pooled balanced accuracy was not reproduced")
    if not np.array_equal(base_pred, np.asarray(LABELS)[fused.argmax(axis=1)]):
        raise AssertionError("Dividing E16 scores by three changed their argmax")
    gate = {
        "pooled_delta_at_least_0_01": bool(new_ba - old_ba >= 0.01 - 1e-12),
        "at_least_four_folds_non_decrease": bool((fold_table["delta"] >= -1e-12).sum() >= 4),
        "all_class_drops_less_than_0_05": bool(class_table["delta"].min() > -0.05 + 1e-12),
    }
    metrics = {
        "experiment": "e23_marginal_correction",
        "status": "completed",
        "input_sha256": hashes,
        "n_rows": len(dev),
        "n_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": N_FOLDS,
        "parameters": {"logit_strength": LOGIT_STRENGTH, "log_ratio_clip": LOG_RATIO_LIMIT,
                       "reference_share": 1.0 / len(LABELS), "base_logits_divisor": 3.0},
        "saved_e16_predictions_exactly_reproduced": True,
        "label_or_prompt_used_in_correction": False,
        "holdout_evaluated": False,
        "weights_tuned": False,
        "e16_balanced_accuracy": old_ba,
        "e23_balanced_accuracy": new_ba,
        "delta": new_ba - old_ba,
        "paired_prompt_bootstrap": paired_prompt_bootstrap(
            true, dev["prompt_id"].to_numpy(), base_pred, new_pred
        ),
        "changed_predictions": int((base_pred != new_pred).sum()),
        "wrong_to_right": int(((base_pred != true) & (new_pred == true)).sum()),
        "right_to_wrong": int(((base_pred == true) & (new_pred != true)).sum()),
        "folds_non_decrease": int((fold_table["delta"] >= -1e-12).sum()),
        "worst_class_delta": float(class_table["delta"].min()),
        "max_abs_logit_shift": float(fold_table["max_abs_logit_shift"].max()),
        "max_pairwise_margin_shift_range": float(fold_table["pairwise_margin_shift_range"].max()),
        "go_criteria": gate,
        "development_go": all(gate.values()),
        "deployment_prior_known": False,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(output_dir / "fold_metrics.csv", index=False)
    class_table.to_csv(output_dir / "class_metrics.csv", index=False)
    marginal_table.to_csv(output_dir / "marginal_metrics.csv", index=False)
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = true
    oof["fold"] = folds
    oof["e16_pred"] = base_pred
    oof["e23_pred"] = new_pred
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
