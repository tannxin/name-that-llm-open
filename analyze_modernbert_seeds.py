#!/usr/bin/env python3
"""Evaluate fixed ModernBERT seed replications on the original development OOF.

The locked holdout is never opened. Each seed uses the same five prompt folds.
Fusion uses the E10 row-z rule without fitted weights; the three-seed ModernBERT
stream first averages the three individually standardized score matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize


MODEL_ID = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
RUN_DIRS = {
    42: Path("outputs/e15_modernbert_base_12ep_eager"),
    17: Path("outputs/e16_modernbert_seed17_eager"),
    73: Path("outputs/e16_modernbert_seed73_eager"),
}
UNCHANGED_CONFIG = (
    "model_id", "requested_revision", "epochs", "encoder_learning_rate",
    "head_learning_rate", "max_length", "batch_size", "warmup_ratio",
    "weight_decay", "dropout", "pooling", "architecture",
    "roberta_pooler_disabled", "train_sha256", "split_manifest_sha256",
    "e0_oof_sha256", "n_rows_development", "n_prompts_development",
    "holdout_prompts_untouched", "epoch_selection",
)


def family(label: str) -> str:
    return "qwen" if label.startswith("qwen") else label.split("-", 1)[0]


def family_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true_family = np.asarray([family(label) for label in true])
    pred_family = np.asarray([family(label) for label in predicted])
    family_hits = true_family == pred_family
    return {
        "family_accuracy": float(family_hits.mean()),
        "exact_given_correct_family": float((true == predicted).sum() / family_hits.sum()),
    }


def score_columns(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)


def validate_keys(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    for key in KEY_COLUMNS:
        if not reference[key].equals(other[key]):
            raise ValueError(f"{name} differs from E0c on {key}")


def validate_run(seed: int, path: Path, reference_config: dict) -> pd.DataFrame:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if config["seed"] != seed or metrics["seed"] != seed:
        raise ValueError(f"Seed {seed} differs from saved run configuration")
    if config["model_id"] != MODEL_ID or config["requested_revision"] != REVISION:
        raise ValueError(f"Unexpected ModernBERT model or revision for seed {seed}")
    if metrics["pretrained_revision"] != REVISION:
        raise ValueError(f"Loaded pretrained revision differs for seed {seed}")
    for key in UNCHANGED_CONFIG:
        if config[key] != reference_config[key]:
            raise ValueError(f"Seed {seed} changed fixed E15 config key {key}")
    frame = load_scores(path / "oof_predictions.csv")
    predicted = np.asarray(LABELS)[score_columns(frame).argmax(axis=1)]
    if not np.array_equal(predicted, frame["pred_model"].to_numpy()):
        raise ValueError(f"Seed {seed} saved predictions disagree with its logits")
    observed_ba = balanced_accuracy_score(frame["true_model"], predicted)
    if not np.isclose(observed_ba, metrics["oof_balanced_accuracy"], atol=1e-12):
        raise ValueError(f"Seed {seed} saved balanced accuracy differs from logits")
    return frame


def run(args: argparse.Namespace) -> None:
    seeds = (42, *args.candidate_seeds)
    if len(set(seeds)) != len(seeds) or not set(args.candidate_seeds).issubset({17, 73}):
        raise ValueError("Candidate seeds must be distinct and drawn from 17 and 73")
    reference_config = json.loads((args.seed42_dir / "run_config.json").read_text(encoding="utf-8"))
    run_dirs = {42: args.seed42_dir, 17: args.seed17_dir, 73: args.seed73_dir}
    modern = {seed: validate_run(seed, run_dirs[seed], reference_config) for seed in seeds}
    e0c = load_scores(args.e0c_scores)
    roberta = load_scores(args.roberta_scores)
    if len(e0c) != 2160 or e0c["prompt_id"].nunique() != 120:
        raise ValueError("Expected the original 2,160-row, 120-prompt development OOF")
    if set(e0c["fold"]) != {str(fold) for fold in range(1, 6)}:
        raise ValueError("Expected five original prompt-grouped folds")
    prompt_labels = e0c.groupby("prompt_id")["true_model"].agg(list)
    if any(len(labels) != len(LABELS) or set(labels) != set(LABELS)
           for labels in prompt_labels):
        raise ValueError("Each development prompt must contain each class once")
    validate_keys(e0c, roberta, "RoBERTa")
    for seed, frame in modern.items():
        validate_keys(e0c, frame, f"ModernBERT seed {seed}")
    published = pd.read_csv(args.e10_oof, dtype=str, keep_default_na=False)
    validate_keys(e0c, published, "Saved E10")

    labels = np.asarray(LABELS)
    true = e0c["true_model"].to_numpy()
    prompts = e0c["prompt_id"].to_numpy()
    ze0c = row_standardize(score_columns(e0c))
    zroberta = row_standardize(score_columns(roberta))
    zmodern = {seed: row_standardize(score_columns(frame)) for seed, frame in modern.items()}
    predictions = {
        "e10": labels[(ze0c + zroberta).argmax(axis=1)],
    }
    if not np.array_equal(predictions["e10"], published["zscore_pred_model"].to_numpy()):
        raise ValueError("Fixed E10 rule did not reproduce saved predictions")
    for seed in seeds:
        modern_direct = labels[score_columns(modern[seed]).argmax(axis=1)]
        predictions[f"modern_seed{seed}"] = modern_direct
        predictions[f"e0c_modern_seed{seed}"] = labels[(ze0c + zmodern[seed]).argmax(axis=1)]
        predictions[f"e10_modern_seed{seed}"] = labels[(ze0c + zroberta + zmodern[seed]).argmax(axis=1)]
    # This independently checks that seed 42 still reproduces the E15 report.
    e15 = json.loads((args.seed42_dir / "three_way_fixed_e10.json").read_text(encoding="utf-8"))
    if not np.isclose(
        balanced_accuracy_score(true, predictions["e10_modern_seed42"]),
        e15["three_way_ba"], atol=1e-12,
    ):
        raise ValueError("Seed 42 three-way score failed to reproduce E15")
    saved_e15 = pd.read_csv(args.seed42_dir / "three_way_fusion_oof.csv", dtype=str)
    validate_keys(e0c, saved_e15, "Saved E15")
    if not np.array_equal(predictions["e10_modern_seed42"], saved_e15["pred_model"].to_numpy()):
        raise ValueError("Seed 42 three-way predictions failed to reproduce E15")

    if set(seeds) == {42, 17, 73}:
        mean_modern = np.mean(np.stack([zmodern[seed] for seed in (42, 17, 73)]), axis=0)
        predictions["modern_three_seed_mean"] = labels[mean_modern.argmax(axis=1)]
        predictions["e0c_modern_three_seed_mean"] = labels[(ze0c + mean_modern).argmax(axis=1)]
        predictions["e10_modern_three_seed_mean"] = labels[(ze0c + zroberta + mean_modern).argmax(axis=1)]

    fold_rows: list[dict[str, object]] = []
    for fold in range(1, 6):
        mask = e0c["fold"].eq(str(fold)).to_numpy()
        fold_rows.append({
            "fold": fold,
            **{name: float(balanced_accuracy_score(true[mask], pred[mask]))
               for name, pred in predictions.items()},
        })
    fold_metrics = pd.DataFrame(fold_rows)
    class_rows: list[dict[str, object]] = []
    for index, label in enumerate(LABELS):
        mask = true == label
        class_rows.append({
            "model": label,
            "support": int(mask.sum()),
            **{name: float((pred[mask] == label).mean())
               for name, pred in predictions.items()},
        })
    class_metrics = pd.DataFrame(class_rows)
    if not (class_metrics["support"] == 120).all():
        raise ValueError("Development class supports are not all 120")
    prompt_rows: list[dict[str, object]] = []
    for prompt_id, group in e0c.groupby("prompt_id", sort=True):
        index = group.index.to_numpy()
        base_correct = int((predictions["e10"][index] == true[index]).sum())
        row: dict[str, object] = {
            "prompt_id": prompt_id,
            "fold": str(group["fold"].iloc[0]),
            "e10_correct": base_correct,
        }
        for name, predicted in predictions.items():
            if name.startswith("e10_modern"):
                correct = int((predicted[index] == true[index]).sum())
                row[f"{name}_correct"] = correct
                row[f"{name}_delta"] = correct - base_correct
        prompt_rows.append(row)
    prompt_metrics = pd.DataFrame(prompt_rows)
    if not (prompt_metrics["e10_correct"].between(0, len(LABELS))).all():
        raise ValueError("Prompt-level counts are invalid")
    metrics = {
        "experiment": "e16_modernbert_seed_replication",
        "seeds": list(seeds),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "fusion_rule": "row-z each source; equal E0c, RoBERTa and ModernBERT streams; for 3 seeds average their row-z scores before adding as one stream",
        "n_rows_development": len(e0c),
        "n_prompts_development": len(prompt_labels),
        "holdout_scored": False,
        "fusion_weights_tuned": False,
        "score_sha256": {
            "e0c": sha256(args.e0c_scores),
            "roberta": sha256(args.roberta_scores),
            "e10_oof": sha256(args.e10_oof),
            **{f"modern_seed{seed}": sha256(run_dirs[seed] / "oof_predictions.csv")
               for seed in seeds},
        },
        "balanced_accuracy": {
            name: float(balanced_accuracy_score(true, pred))
            for name, pred in predictions.items()
        },
        "family": {name: family_metrics(true, pred)
                   for name, pred in predictions.items()},
        "comparisons_vs_e10": {
            name: paired_prompt_bootstrap(true, prompts, predictions["e10"], pred)
            for name, pred in predictions.items() if name.startswith("e10_modern")
        },
        "fold_direction_vs_e10": {
            name: {
                "higher": int((fold_metrics[name] > fold_metrics["e10"] + 1e-12).sum()),
                "equal": int(np.isclose(fold_metrics[name], fold_metrics["e10"], atol=1e-12).sum()),
                "lower": int((fold_metrics[name] < fold_metrics["e10"] - 1e-12).sum()),
            }
            for name in predictions if name.startswith("e10_modern")
        },
        "wrong_to_right_vs_e10": {
            name: int(((predictions["e10"] != true) & (pred == true)).sum())
            for name, pred in predictions.items() if name.startswith("e10_modern")
        },
        "right_to_wrong_vs_e10": {
            name: int(((predictions["e10"] == true) & (pred != true)).sum())
            for name, pred in predictions.items() if name.startswith("e10_modern")
        },
        "prompt_directions_vs_e10": {
            name: {
                "improved": int((prompt_metrics[f"{name}_delta"] > 0).sum()),
                "equal": int((prompt_metrics[f"{name}_delta"] == 0).sum()),
                "harmed": int((prompt_metrics[f"{name}_delta"] < 0).sum()),
                "worst_prompt_id": str(prompt_metrics.loc[
                    prompt_metrics[f"{name}_delta"].idxmin(), "prompt_id"
                ]),
                "worst_prompt_delta": int(prompt_metrics[f"{name}_delta"].min()),
            }
            for name in predictions if name.startswith("e10_modern")
        },
    }
    if set(seeds) == {42, 17, 73}:
        seed_delta_cols = [f"e10_modern_seed{seed}_delta" for seed in (42, 17, 73)]
        delta = prompt_metrics[seed_delta_cols]
        ensemble_delta = prompt_metrics["e10_modern_three_seed_mean_delta"]
        metrics["shared_prompt_patterns"] = {
            "all_three_seed_improved": int((delta > 0).all(axis=1).sum()),
            "all_three_seed_harmed": int((delta < 0).all(axis=1).sum()),
            "ensemble_harmed_and_all_three_seed_harmed": int(
                ((ensemble_delta < 0) & (delta < 0).all(axis=1)).sum()
            ),
            "ensemble_harmed_and_at_least_two_seed_harmed": int(
                ((ensemble_delta < 0) & ((delta < 0).sum(axis=1) >= 2)).sum()
            ),
            "seed_prompt_delta_correlations": {
                f"{left}_{right}": float(delta.iloc[:, i].corr(delta.iloc[:, j]))
                for i, left in enumerate((42, 17, 73))
                for j, right in enumerate((42, 17, 73)) if i < j
            },
        }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    prompt_metrics.to_csv(args.output_dir / "prompt_metrics.csv", index=False)
    oof = e0c.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in predictions.items():
        oof[f"pred_{name}"] = pred
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-seeds", type=int, nargs="*", default=[17, 73])
    parser.add_argument("--seed42-dir", type=Path, default=RUN_DIRS[42])
    parser.add_argument("--seed17-dir", type=Path, default=RUN_DIRS[17])
    parser.add_argument("--seed73-dir", type=Path, default=RUN_DIRS[73])
    parser.add_argument("--e0c-scores", type=Path,
                        default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    parser.add_argument("--roberta-scores", type=Path,
                        default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    parser.add_argument("--e10-oof", type=Path,
                        default=Path("outputs/e10_fusion_normalization/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/e16_modernbert_seed_analysis"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
