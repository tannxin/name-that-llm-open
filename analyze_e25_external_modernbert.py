#!/usr/bin/env python3
"""Compare locked E25 five-fold OOF streams against unchanged E16 fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from analyze_modernbert_seeds import family_metrics, score_columns, validate_keys, validate_run
from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize


def load_candidate(path: Path, recipe: str, seed: int, reference: pd.DataFrame) -> pd.DataFrame:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    required = {
        "experiment": "e25_external_modernbert", "stage": "fit",
        "recipe": recipe, "seed": seed,
        "model_id": "answerdotai/ModernBERT-base",
        "requested_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "max_length": 256, "batch_size": 16,
        "encoder_lr": 2e-5, "head_lr": 1e-3,
        "official_train_sha256": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"E25 {recipe} seed {seed} changed {key}")
    if metrics["recipe"] != recipe or metrics["seed"] != seed:
        raise ValueError("E25 metrics recipe/seed mismatch")
    if len(metrics["fold_metrics"]) != 5:
        raise ValueError("E25 run is not complete five-fold OOF")
    if config["source_protocol_uses_official_global_length_statistics"] is not True:
        raise ValueError("E25 source limitation missing")
    frame = load_scores(path / "oof_predictions.csv")
    validate_keys(reference, frame, f"E25 {recipe} seed {seed}")
    predicted = np.asarray(LABELS)[score_columns(frame).argmax(axis=1)]
    if not np.array_equal(predicted, frame["pred_model"].to_numpy()):
        raise ValueError(f"E25 {recipe} saved argmax mismatch")
    observed = balanced_accuracy_score(frame["true_model"], predicted)
    if not np.isclose(observed, metrics["oof_balanced_accuracy"], atol=1e-12):
        raise ValueError(f"E25 {recipe} saved BA mismatch")
    return frame


def compare(args):
    reference_config = json.loads((args.seed42_dir / "run_config.json").read_text(encoding="utf-8"))
    modern = {
        42: validate_run(42, args.seed42_dir, reference_config),
        17: validate_run(17, args.seed17_dir, reference_config),
        73: validate_run(73, args.seed73_dir, reference_config),
    }
    e0c = load_scores(args.e0c_scores)
    e24 = load_scores(args.e24_scores)
    roberta = load_scores(args.roberta_scores)
    for name, frame in [("E24", e24), ("RoBERTa", roberta),
                        *[(f"ModernBERT {seed}", modern[seed]) for seed in (42, 17, 73)]]:
        validate_keys(e0c, frame, name)
    if len(e0c) != 2160 or e0c["prompt_id"].nunique() != 120:
        raise ValueError("Expected 2,160 rows and 120 development prompts")
    true = e0c["true_model"].to_numpy()
    prompts = e0c["prompt_id"].to_numpy()
    folds = e0c["fold"].to_numpy()
    labels = np.asarray(LABELS)
    if args.seed not in (42, 17):
        raise ValueError("Only pre-registered seed 42/17 analyses allowed")
    # The pre-registered seed-17 replication runs only for recipes that passed
    # the seed-42 gate. C failed, so its nonexistent seed-17 run is not loaded.
    candidates = {"B": load_candidate(args.recipe_b_dir, "B", args.seed, e0c)}
    if args.seed == 42:
        candidates["C"] = load_candidate(args.recipe_c_dir, "C", args.seed, e0c)
    seed42_b = (load_candidate(args.recipe_b_seed42_dir, "B", 42, e0c)
                if args.seed == 17 else None)
    old_z = {seed: row_standardize(score_columns(frame)) for seed, frame in modern.items()}
    old_mean = np.mean(np.stack([old_z[seed] for seed in (42, 17, 73)]), axis=0)
    e0c_z = row_standardize(score_columns(e0c))
    e24_z = row_standardize(score_columns(e24))
    roberta_z = row_standardize(score_columns(roberta))
    original_fused = labels[(e0c_z + roberta_z + old_mean).argmax(axis=1)]
    saved_e16 = pd.read_csv(args.e16_oof, dtype=str, keep_default_na=False)
    validate_keys(e0c, saved_e16, "Saved E16")
    if not np.array_equal(original_fused,
                          saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("Original E16 row-level predictions failed to reproduce")
    if not np.isclose(balanced_accuracy_score(true, original_fused),
                      0.40925925925925927, atol=1e-12):
        raise ValueError("Original E16 BA failed to reproduce")
    e24_fused = labels[(e24_z + roberta_z + old_mean).argmax(axis=1)]
    saved_e24 = pd.read_csv(args.e24_oof, dtype=str, keep_default_na=False)
    validate_keys(e0c, saved_e24, "Saved E24 fusion")
    if not np.array_equal(e24_fused, saved_e24["pred_e16_with_e24"].to_numpy()):
        raise ValueError("E24 secondary baseline failed to reproduce")

    pred = {
        "official_seed_comparator": labels[score_columns(modern[args.seed]).argmax(axis=1)],
        "e16_main": original_fused,
        "e16_e24_secondary": e24_fused,
    }
    for recipe, frame in candidates.items():
        new_z = row_standardize(score_columns(frame))
        replacement = dict(old_z)
        replacement[args.seed] = new_z
        mean = np.mean(np.stack([replacement[seed] for seed in (42, 17, 73)]), axis=0)
        pred[f"{recipe}_single"] = labels[score_columns(frame).argmax(axis=1)]
        pred[f"{recipe}_main"] = labels[(e0c_z + roberta_z + mean).argmax(axis=1)]
        pred[f"{recipe}_secondary"] = labels[(e24_z + roberta_z + mean).argmax(axis=1)]
    if seed42_b is not None:
        both = dict(old_z)
        both[42] = row_standardize(score_columns(seed42_b))
        both[17] = row_standardize(score_columns(candidates["B"]))
        both_mean = np.mean(np.stack([both[seed] for seed in (42, 17, 73)]), axis=0)
        pred["B_both_seeds_main"] = labels[(e0c_z + roberta_z + both_mean).argmax(axis=1)]
        pred["B_both_seeds_secondary"] = labels[(e24_z + roberta_z + both_mean).argmax(axis=1)]
    score = {name: float(balanced_accuracy_score(true, value)) for name, value in pred.items()}
    fold_rows = []
    for fold in range(1, 6):
        mask = folds == str(fold)
        row = {"fold": fold, **{name: float(balanced_accuracy_score(true[mask], value[mask]))
                               for name, value in pred.items()}}
        for recipe in candidates:
            row[f"{recipe}_single_delta"] = row[f"{recipe}_single"] - row["official_seed_comparator"]
            row[f"{recipe}_main_delta"] = row[f"{recipe}_main"] - row["e16_main"]
            row[f"{recipe}_secondary_delta"] = row[f"{recipe}_secondary"] - row["e16_e24_secondary"]
        fold_rows.append(row)
    fold_frame = pd.DataFrame(fold_rows)
    class_rows = []
    for label in LABELS:
        mask = true == label
        row = {"model": label, "support": int(mask.sum()),
               **{name: float((value[mask] == label).mean()) for name, value in pred.items()}}
        for recipe in candidates:
            row[f"{recipe}_single_delta"] = row[f"{recipe}_single"] - row["official_seed_comparator"]
            row[f"{recipe}_main_delta"] = row[f"{recipe}_main"] - row["e16_main"]
            row[f"{recipe}_secondary_delta"] = row[f"{recipe}_secondary"] - row["e16_e24_secondary"]
        class_rows.append(row)
    class_frame = pd.DataFrame(class_rows)
    if not (class_frame["support"] == 120).all():
        raise ValueError("Official class supports changed")
    gate = {}
    for recipe in candidates:
        delta = score[f"{recipe}_main"] - score["e16_main"]
        nondecrease = int((fold_frame[f"{recipe}_main_delta"] >= -1e-12).sum())
        worst = float(class_frame[f"{recipe}_main_delta"].min())
        if args.seed == 42:
            gate[recipe] = {
                "fusion_delta_at_least_0_01": delta >= 0.01 - 1e-12,
                "at_least_four_folds_non_decreasing": nondecrease >= 4,
                "no_class_drop_at_least_0_05": worst > -0.05 + 1e-12,
                "passed": delta >= 0.01 - 1e-12 and nondecrease >= 4
                          and worst > -0.05 + 1e-12,
                "fusion_delta": delta, "folds_non_decreasing": nondecrease,
                "worst_class_delta": worst,
            }
        else:
            gate[recipe] = {
                "fusion_delta_positive": delta > 0,
                "at_least_four_folds_non_decreasing": nondecrease >= 4,
                "no_class_drop_at_least_0_05": worst > -0.05 + 1e-12,
                "passed": delta > 0 and nondecrease >= 4 and worst > -0.05 + 1e-12,
                "fusion_delta": delta, "folds_non_decreasing": nondecrease,
                "worst_class_delta": worst,
            }
    metrics = {
        "experiment": "e25_external_modernbert", "analysis_seed": args.seed,
        "rows": len(e0c), "prompts": len(np.unique(prompts)),
        "original_e16_predictions_reproduced": True,
        "e24_secondary_predictions_reproduced": True,
        "sources_sha256": {
            "e0c": sha256(args.e0c_scores), "e24": sha256(args.e24_scores),
            "roberta": sha256(args.roberta_scores),
            **{f"official_seed{seed}": sha256(path / "oof_predictions.csv")
               for seed, path in ((42, args.seed42_dir), (17, args.seed17_dir),
                                  (73, args.seed73_dir))},
            **{f"candidate_{recipe}": sha256(path / "oof_predictions.csv")
               for recipe, path in (("B", args.recipe_b_dir), ("C", args.recipe_c_dir))
               if recipe in candidates},
            **({"candidate_B_seed42": sha256(args.recipe_b_seed42_dir / "oof_predictions.csv")}
               if seed42_b is not None else {}),
        },
        "balanced_accuracy": score,
        "family_metrics": {name: family_metrics(true, value) for name, value in pred.items()},
        "paired_prompt_bootstrap": {
            recipe: {
                "single_vs_official_seed": paired_prompt_bootstrap(
                    true, prompts, pred["official_seed_comparator"], pred[f"{recipe}_single"]),
                "main_vs_e16": paired_prompt_bootstrap(
                    true, prompts, pred["e16_main"], pred[f"{recipe}_main"]),
                "secondary_vs_e24": paired_prompt_bootstrap(
                    true, prompts, pred["e16_e24_secondary"], pred[f"{recipe}_secondary"]),
            } for recipe in candidates
        },
        "gate": gate,
        "fusion_weights_tuned": False,
        "source_protocol_uses_official_global_length_statistics": True,
        "holdout_evaluated": False,
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output dir already contains files: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_frame.to_csv(args.output_dir / "class_metrics.csv", index=False)
    out = e0c.loc[:, list(KEY_COLUMNS)].copy()
    for name, values in pred.items():
        out[f"pred_{name}"] = values
    out.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(json.dumps({"balanced_accuracy": score, "gate": gate}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, choices=(42, 17))
    parser.add_argument("--recipe-b-dir", type=Path,
                        default=Path("outputs/e25_modernbert_B_seed42"))
    parser.add_argument("--recipe-c-dir", type=Path,
                        default=Path("outputs/e25_modernbert_C_seed42"))
    parser.add_argument("--recipe-b-seed42-dir", type=Path,
                        default=Path("outputs/e25_modernbert_B_seed42"))
    parser.add_argument("--e0c-scores", type=Path,
                        default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    parser.add_argument("--e24-scores", type=Path,
                        default=Path("outputs/e24_prvsiyan_fit/decision_scores.csv"))
    parser.add_argument("--roberta-scores", type=Path,
                        default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    parser.add_argument("--seed42-dir", type=Path,
                        default=Path("outputs/e15_modernbert_base_12ep_eager"))
    parser.add_argument("--seed17-dir", type=Path,
                        default=Path("outputs/e16_modernbert_seed17_eager"))
    parser.add_argument("--seed73-dir", type=Path,
                        default=Path("outputs/e16_modernbert_seed73_eager"))
    parser.add_argument("--e16-oof", type=Path,
                        default=Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv"))
    parser.add_argument("--e24-oof", type=Path,
                        default=Path("outputs/e24_prvsiyan_analysis/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/e25_modernbert_analysis_seed42"))
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
