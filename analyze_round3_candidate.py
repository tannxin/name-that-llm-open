#!/usr/bin/env python3
"""Audit one fixed RoBERTa OOF candidate against E2d-R and E10.

The original prompt folds and E10 fusion rule are checked row for row. No
parameter is selected on these predictions, and the locked holdout is unused.
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
from run_score_normalization import paired_prompt_bootstrap, row_standardize


def family(label: str) -> str:
    return "qwen" if label.startswith("qwen") else label.split("-", 1)[0]


def family_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    family_hit = np.asarray([family(a) == family(b) for a, b in zip(true, predicted)])
    exact_hit = true == predicted
    return {
        "family_accuracy": float(family_hit.mean()),
        "exact_given_correct_family": float(exact_hit.sum() / family_hit.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--lengths", type=Path, help="Development-only id,word_count,token_count CSV")
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    parser.add_argument("--e0c", type=Path, default=Path(
        "outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    parser.add_argument("--e10", type=Path, default=Path(
        "outputs/e10_fusion_normalization/oof_predictions.csv"))
    args = parser.parse_args()
    original_config = json.loads((args.baseline.parent / "run_config.json").read_text())
    candidate_config = json.loads((args.candidate.parent / "run_config.json").read_text())
    for key in (
        "model_id", "requested_revision", "epochs", "encoder_learning_rate",
        "head_learning_rate", "max_length", "batch_size", "seed", "warmup_ratio",
        "weight_decay", "dropout", "pooling", "train_sha256",
        "split_manifest_sha256", "e0_oof_sha256", "n_rows_development",
        "n_prompts_development", "epoch_selection",
    ):
        if candidate_config[key] != original_config[key]:
            raise ValueError(f"Candidate differs from E2d-R in fixed setting: {key}")
    if candidate_config.get("holdout_scored") is not False:
        raise ValueError("Candidate config does not affirm holdout was unscored")

    e0c, original, candidate = map(load_scores, (args.e0c, args.baseline, args.candidate))
    if len(e0c) != 2160 or e0c["prompt_id"].nunique() != 120:
        raise ValueError("Expected 120 development prompts and 2160 rows")
    for other in (original, candidate):
        for key in KEY_COLUMNS:
            if not e0c[key].equals(other[key]):
                raise ValueError(f"OOF {key} rows do not align exactly")
    if set(e0c["fold"]) != {str(x) for x in range(1, 6)}:
        raise ValueError("Expected five original prompt folds")
    reference = pd.read_csv(args.e10, dtype={key: str for key in KEY_COLUMNS})
    for key in KEY_COLUMNS:
        if not e0c[key].equals(reference[key]):
            raise ValueError(f"Published E10 {key} rows differ")

    labels = np.asarray(LABELS)
    true = e0c["true_model"].to_numpy()
    prompts = e0c["prompt_id"].to_numpy()
    fold = e0c["fold"].to_numpy()
    z0 = row_standardize(e0c.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=float))
    base_scores = original.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=float)
    new_scores = candidate.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=float)
    base_direct = labels[base_scores.argmax(axis=1)]
    new_direct = labels[new_scores.argmax(axis=1)]
    base_fused = labels[(z0 + row_standardize(base_scores)).argmax(axis=1)]
    new_fused = labels[(z0 + row_standardize(new_scores)).argmax(axis=1)]
    if not np.array_equal(base_direct, original["pred_model"].to_numpy()):
        raise ValueError("Baseline argmax does not equal saved prediction")
    if not np.array_equal(new_direct, candidate["pred_model"].to_numpy()):
        raise ValueError("Candidate argmax does not equal saved prediction")
    if not np.array_equal(base_fused, reference["zscore_pred_model"].to_numpy()):
        raise ValueError("Fixed fusion does not reproduce E10 exactly")

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing analysis")
    output.mkdir(parents=True)
    cases = {
        "e2dr_direct": base_direct, "candidate_direct": new_direct,
        "e10_fused": base_fused, "candidate_fused": new_fused,
    }
    metrics = {
        "candidate_oof_sha256": sha256(args.candidate),
        "baseline_oof_sha256": sha256(args.baseline),
        "e0c_scores_sha256": sha256(args.e0c),
        "published_e10_sha256": sha256(args.e10),
        "n_rows": len(true), "n_prompts": len(np.unique(prompts)),
        "balanced_accuracy": {name: float(balanced_accuracy_score(true, pred))
                              for name, pred in cases.items()},
        "family_metrics": {name: family_metrics(true, pred)
                           for name, pred in cases.items()},
        "paired_prompt_bootstrap_direct": paired_prompt_bootstrap(
            true, prompts, base_direct, new_direct),
        "paired_prompt_bootstrap_fused": paired_prompt_bootstrap(
            true, prompts, base_fused, new_fused),
        "fused_switches": {
            "wrong_to_right": int(((base_fused != true) & (new_fused == true)).sum()),
            "right_to_wrong": int(((base_fused == true) & (new_fused != true)).sum()),
        },
    }
    fold_rows = []
    for index in range(1, 6):
        mask = fold == str(index)
        fold_rows.append({"fold": index, **{
            name: float(balanced_accuracy_score(true[mask], pred[mask]))
            for name, pred in cases.items()
        }})
    metrics["fold_metrics"] = fold_rows
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)

    class_table = pd.DataFrame({"model": LABELS, "support": [120] * len(LABELS)})
    for name, pred in cases.items():
        matrix = confusion_matrix(true, pred, labels=LABELS)
        class_table[f"{name}_recall"] = matrix.diagonal() / matrix.sum(axis=1)
    class_table["fused_recall_difference"] = (
        class_table["candidate_fused_recall"] - class_table["e10_fused_recall"])
    class_table.to_csv(output / "class_metrics.csv", index=False)

    if args.selection_config:
        selection = json.loads(args.selection_config.read_text())
        if selection["n_inner_oof_rows"] != 5 * 1728 or selection["n_pairs"] != 2:
            raise ValueError("Unexpected margin selection manifest")
        target_rows = []
        for entry in selection["selected_by_outer_fold"]:
            outer = int(entry["outer_fold"])
            for left, right in entry["selected_pairs"]:
                for label, partner in ((left, right), (right, left)):
                    mask = (fold == str(outer)) & (true == label)
                    target_rows.append({
                        "fold": outer, "label": label, "partner": partner,
                        "n": int(mask.sum()),
                        "e10_recall": float((base_fused[mask] == label).mean()),
                        "candidate_recall": float((new_fused[mask] == label).mean()),
                        "e10_to_partner": int((base_fused[mask] == partner).sum()),
                        "candidate_to_partner": int((new_fused[mask] == partner).sum()),
                    })
        metrics["selection_config_sha256"] = sha256(args.selection_config)
        pd.DataFrame(target_rows).to_csv(output / "target_pair_metrics.csv", index=False)

    if args.lengths:
        lengths = pd.read_csv(args.lengths, dtype={"id": str})
        if set(lengths.columns) != {"id", "word_count", "token_count"}:
            raise ValueError("Expected development-only id,word_count,token_count lengths")
        if lengths["id"].duplicated().any() or set(lengths["id"]) != set(e0c["id"]):
            raise ValueError("Length rows differ from OOF development IDs")
        lengths = e0c[["id"]].merge(lengths, on="id", validate="one_to_one")
        strata = {}
        for key, mask in (
            ("words_under_120", lengths["word_count"].to_numpy() < 120),
            ("words_120_plus", lengths["word_count"].to_numpy() >= 120),
            ("tokens_256_or_less", lengths["token_count"].to_numpy() <= 256),
            ("tokens_over_256", lengths["token_count"].to_numpy() > 256),
        ):
            strata[key] = {"n": int(mask.sum()), **{
                name + "_accuracy": float((pred[mask] == true[mask]).mean())
                for name, pred in cases.items()
            }}
        metrics["length_strata"] = strata

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"balanced_accuracy": metrics["balanced_accuracy"],
                      "paired_prompt_bootstrap_fused": metrics["paired_prompt_bootstrap_fused"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
