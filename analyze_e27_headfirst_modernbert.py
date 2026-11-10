#!/usr/bin/env python3
"""Analyze preregistered E27 2x2 OOF and unchanged E24 fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from analyze_e25_external_modernbert import load_candidate as load_e25
from analyze_modernbert_seeds import (
    family_metrics, score_columns, validate_keys, validate_run,
)
from run_e0 import LABELS, sha256
from run_fixed_blend import KEY_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize


def load_e27(path: Path, recipe: str, reference: pd.DataFrame) -> pd.DataFrame:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    required = {
        "experiment": "e27_headfirst_modernbert", "stage": "fit",
        "recipe": recipe, "seed": 42,
        "model_id": "answerdotai/ModernBERT-base",
        "requested_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "head_epochs": 2, "top_epochs": 10, "top_encoder_blocks": 2,
        "max_length": 256, "batch_size": 16,
        "encoder_lr": 2e-5, "head_lr": 1e-3,
        "official_train_sha256": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
        "selected_external_sha256": "d7f8b7c9b0480ec896777e667589bf7d2fe12e503aa8f867c0708d0656ad6191",
        "source_protocol_uses_official_global_length_statistics": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"E27 {recipe} unexpected {key}")
    if metrics["recipe"] != recipe or metrics["seed"] != 42:
        raise ValueError(f"E27 {recipe} metrics recipe/seed mismatch")
    if len(metrics["fold_metrics"]) != 5:
        raise ValueError(f"E27 {recipe} five folds incomplete")
    if any(item["external_train_rows"] != (0 if recipe == "O" else 3620)
           for item in metrics["fold_metrics"]):
        raise ValueError(f"E27 {recipe} external rows changed")
    frame = load_scores(path / "oof_predictions.csv")
    validate_keys(reference, frame, f"E27 {recipe}")
    predicted = np.asarray(LABELS)[score_columns(frame).argmax(axis=1)]
    if not np.array_equal(predicted, frame["pred_model"].to_numpy()):
        raise ValueError(f"E27 {recipe} argmax mismatch")
    observed = balanced_accuracy_score(frame["true_model"], predicted)
    if not np.isclose(observed, metrics["oof_balanced_accuracy"], atol=1e-12):
        raise ValueError(f"E27 {recipe} metrics BA mismatch")
    return frame


def paired_interaction(true, prompts, four_predictions):
    """Prompt bootstrap of (X-B)-(O-A); each prompt has one of each class."""
    names = ("A", "B", "O", "X")
    prompt_ids = np.unique(prompts)
    per_prompt = {}
    for name in names:
        correct = (four_predictions[name] == true).astype(float)
        per_prompt[name] = np.asarray([
            correct[prompts == prompt].mean() for prompt in prompt_ids
        ])
    deltas = (per_prompt["X"] - per_prompt["B"]
              - per_prompt["O"] + per_prompt["A"])
    rng = np.random.default_rng(42)
    bootstrap = np.empty(20000, dtype=float)
    for i in range(len(bootstrap)):
        sample = rng.integers(0, len(prompt_ids), len(prompt_ids))
        bootstrap[i] = deltas[sample].mean()
    return {
        "delta": float(deltas.mean()),
        "ci_low": float(np.percentile(bootstrap, 2.5)),
        "ci_high": float(np.percentile(bootstrap, 97.5)),
        "unit": "official development prompt",
        "resamples": 20000,
    }


def analyze(args):
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
                        *[(f"ModernBERT seed {seed}", modern[seed])
                          for seed in (42, 17, 73)]]:
        validate_keys(e0c, frame, name)
    if len(e0c) != 2160 or e0c["prompt_id"].nunique() != 120:
        raise ValueError("Official OOF must have 2,160 rows and 120 prompts")
    if any(len(group) != 18 or set(group) != set(LABELS)
           for _, group in e0c.groupby("prompt_id")["true_model"]):
        raise ValueError("Each prompt must contain each of 18 labels")
    b = load_e25(args.e25_b_dir, "B", 42, e0c)
    o = load_e27(args.e27_o_dir, "O", e0c)
    x = load_e27(args.e27_x_dir, "X", e0c)
    labels = np.asarray(LABELS)
    true = e0c["true_model"].to_numpy()
    prompts = e0c["prompt_id"].to_numpy()
    folds = e0c["fold"].to_numpy()
    z_old = {seed: row_standardize(score_columns(modern[seed]))
             for seed in (42, 17, 73)}
    z_e0c = row_standardize(score_columns(e0c))
    z_e24 = row_standardize(score_columns(e24))
    z_roberta = row_standardize(score_columns(roberta))
    def fused(z_sparse, z_seed42):
        z_modern = np.mean(np.stack((z_seed42, z_old[17], z_old[73])), axis=0)
        return labels[(z_sparse + z_roberta + z_modern).argmax(axis=1)]

    e16 = fused(z_e0c, z_old[42])
    saved_e16 = pd.read_csv(args.e16_oof, dtype=str, keep_default_na=False)
    validate_keys(e0c, saved_e16, "Saved E16")
    if not np.array_equal(e16,
                          saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("E16 predictions failed to reproduce")
    if not np.isclose(balanced_accuracy_score(true, e16), 0.40925925925925927):
        raise ValueError("E16 balanced accuracy failed to reproduce")
    e24_base = fused(z_e24, z_old[42])
    saved_e24 = pd.read_csv(args.e24_oof, dtype=str, keep_default_na=False)
    validate_keys(e0c, saved_e24, "Saved E24")
    if not np.array_equal(e24_base, saved_e24["pred_e16_with_e24"].to_numpy()):
        raise ValueError("E24 predictions failed to reproduce")
    if not np.isclose(balanced_accuracy_score(true, e24_base), 0.425):
        raise ValueError("E24 balanced accuracy failed to reproduce")
    streams = {"A": modern[42], "B": b, "O": o, "X": x}
    single = {name: labels[score_columns(frame).argmax(axis=1)]
              for name, frame in streams.items()}
    fusion = {name: fused(z_e24, row_standardize(score_columns(frame)))
              for name, frame in streams.items()}
    if not np.array_equal(fusion["A"], e24_base):
        raise ValueError("E24 baseline changed during 2x2 fusion")
    predictions = {
        "E16": e16,
        **{f"single_{name}": pred for name, pred in single.items()},
        **{f"fused_{name}": pred for name, pred in fusion.items()},
    }
    ba = {name: float(balanced_accuracy_score(true, pred))
          for name, pred in predictions.items()}
    fold_rows = []
    for fold in range(1, 6):
        mask = folds == str(fold)
        row = {"fold": fold, **{
            name: float(balanced_accuracy_score(true[mask], pred[mask]))
            for name, pred in predictions.items()}}
        row["O_vs_A_fused"] = row["fused_O"] - row["fused_A"]
        row["X_vs_B_fused"] = row["fused_X"] - row["fused_B"]
        row["X_vs_A_fused"] = row["fused_X"] - row["fused_A"]
        row["interaction_fused"] = row["X_vs_B_fused"] - row["O_vs_A_fused"]
        fold_rows.append(row)
    fold_frame = pd.DataFrame(fold_rows)
    class_rows = []
    for label in LABELS:
        mask = true == label
        row = {"model": label, "support": int(mask.sum()), **{
            name: float((pred[mask] == label).mean())
            for name, pred in predictions.items()}}
        row["O_vs_A_fused"] = row["fused_O"] - row["fused_A"]
        row["X_vs_B_fused"] = row["fused_X"] - row["fused_B"]
        row["X_vs_A_fused"] = row["fused_X"] - row["fused_A"]
        row["interaction_fused"] = row["X_vs_B_fused"] - row["O_vs_A_fused"]
        class_rows.append(row)
    class_frame = pd.DataFrame(class_rows)
    if not (class_frame["support"] == 120).all():
        raise ValueError("Official class supports changed")
    gates = {}
    for recipe in ("O", "X"):
        delta = ba[f"fused_{recipe}"] - ba["fused_A"]
        nondecrease = int((fold_frame[f"fused_{recipe}"]
                           >= fold_frame["fused_A"] - 1e-12).sum())
        worst = float((class_frame[f"fused_{recipe}"]
                       - class_frame["fused_A"]).min())
        gates[recipe] = {
            "fused_delta_vs_E24": delta,
            "folds_non_decreasing": nondecrease,
            "worst_class_delta": worst,
            "passed": bool(delta >= 0.01 - 1e-12
                           and nondecrease >= 4 and worst > -0.05 + 1e-12),
        }
    bootstrap = {
        "single_O_vs_A": paired_prompt_bootstrap(true, prompts, single["A"], single["O"]),
        "single_X_vs_B": paired_prompt_bootstrap(true, prompts, single["B"], single["X"]),
        "fused_O_vs_A": paired_prompt_bootstrap(true, prompts, fusion["A"], fusion["O"]),
        "fused_X_vs_B": paired_prompt_bootstrap(true, prompts, fusion["B"], fusion["X"]),
        "fused_X_vs_A": paired_prompt_bootstrap(true, prompts, fusion["A"], fusion["X"]),
        "single_interaction": paired_interaction(true, prompts, single),
        "fused_interaction": paired_interaction(true, prompts, fusion),
    }
    metrics = {
        "experiment": "e27_headfirst_modernbert",
        "rows": len(e0c), "prompts": len(np.unique(prompts)),
        "e16_predictions_reproduced": True,
        "e24_predictions_reproduced": True,
        "balanced_accuracy": ba,
        "single_interaction": (ba["single_X"] - ba["single_B"]
                               - ba["single_O"] + ba["single_A"]),
        "fused_interaction": (ba["fused_X"] - ba["fused_B"]
                              - ba["fused_O"] + ba["fused_A"]),
        "family_metrics": {name: family_metrics(true, pred)
                           for name, pred in predictions.items()},
        "paired_prompt_bootstrap": bootstrap,
        "gate": gates,
        "fusion_weights_tuned": False,
        "source_protocol_uses_official_global_length_statistics": True,
        "holdout_evaluated": False,
        "sources_sha256": {
            "E15": sha256(args.seed42_dir / "oof_predictions.csv"),
            "E25_B": sha256(args.e25_b_dir / "oof_predictions.csv"),
            "E27_O": sha256(args.e27_o_dir / "oof_predictions.csv"),
            "E27_X": sha256(args.e27_x_dir / "oof_predictions.csv"),
            "E24_E0c": sha256(args.e24_scores),
            "RoBERTa": sha256(args.roberta_scores),
        },
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory already contains files: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_frame.to_csv(args.output_dir / "class_metrics.csv", index=False)
    out = e0c.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in predictions.items():
        out[f"pred_{name}"] = pred
    out.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(json.dumps({"balanced_accuracy": ba, "gate": gates,
                      "single_interaction": metrics["single_interaction"],
                      "fused_interaction": metrics["fused_interaction"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--e25-b-dir", type=Path,
                        default=Path("outputs/e25_modernbert_B_seed42"))
    parser.add_argument("--e27-o-dir", type=Path,
                        default=Path("outputs/e27_headfirst_O_seed42"))
    parser.add_argument("--e27-x-dir", type=Path,
                        default=Path("outputs/e27_headfirst_X_seed42"))
    parser.add_argument("--e16-oof", type=Path,
                        default=Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv"))
    parser.add_argument("--e24-oof", type=Path,
                        default=Path("outputs/e24_prvsiyan_analysis/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/e27_headfirst_analysis_seed42"))
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
