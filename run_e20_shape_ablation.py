#!/usr/bin/env python3
"""E20: one fixed shape-text block added to E0c, then fixed E16 fusion.

Run ``fit`` on the original experiment server (Python 3.10.12, sklearn 1.7.2).
Run ``analyze`` where the three saved ModernBERT OOF files are available.
No hyperparameter or fusion weight is selected from the validation labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_style import load_e0_split


SHAPE_PATTERN = re.compile(r"[^\W\d_]+")
SCORE_PATH = Path("outputs/e0c_tfidf_plus_style/decision_scores.csv")
PREDICTION_PATH = Path("outputs/e0c_tfidf_plus_style/oof_predictions.csv")
E16_PATH = Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv")
SEED_DIRS = {
    42: Path("outputs/e15_modernbert_base_12ep_eager"),
    17: Path("outputs/e16_modernbert_seed17_eager"),
    73: Path("outputs/e16_modernbert_seed73_eager"),
}


def shape_texts(texts: np.ndarray) -> list[str]:
    # Exact V47 transform: m[0].isupper() tests the matched word, not just its
    # first letter. All digits, punctuation, underscores and spacing survive.
    return [SHAPE_PATTERN.sub(lambda m: "X" if m[0].isupper() else "x", text)
            for text in texts]


def make_shape_pipeline() -> Pipeline:
    model = make_pipeline()
    features = model.named_steps["features"]
    if [name for name, _ in features.transformer_list] != ["word", "char", "style"]:
        raise ValueError("Original E0c feature blocks changed")
    features.transformer_list.append(("shape", Pipeline([
        ("extract", FunctionTransformer(shape_texts, validate=False)),
        ("tfidf", TfidfVectorizer(
            analyzer="char", ngram_range=(2, 6), lowercase=False,
            sublinear_tf=True, min_df=2, max_features=60_000,
            dtype=np.float32,
        )),
    ])))
    features.transformer_weights = {"shape": 0.5}
    return model


def ordered_scores(model: Pipeline, texts: np.ndarray) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_
    positions = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    return model.decision_function(texts)[:, positions].astype(np.float32)


def fit(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    frame = load_and_validate(args.train)
    dev, _, e0, folds, _ = load_e0_split(frame, args.e0_dir, args.train)
    expected = pd.read_csv(args.e0c_predictions, dtype=str, keep_default_na=False)
    original_scores = load_scores(args.e0c_scores)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Expected 2,160 development rows and 120 prompts")
    for key in KEY_COLUMNS:
        if not expected[key].equals(original_scores[key]) or not expected[key].equals(e0[key]):
            raise ValueError(f"Saved E0c/E0 rows do not align on {key}")
    if sha256(args.train) != "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707":
        raise ValueError("Official train.csv differs from the original E0c run")
    if sklearn.__version__ != "1.7.2" or sys.version_info[:3] != (3, 10, 12):
        raise ValueError("Fit must run in original Python 3.10.12 / sklearn 1.7.2 environment")
    if shape_texts(np.array(["Hello WORLD x3 _keep.\nAa"])) != ["x X x3 _x.\nx"]:
        raise AssertionError("Shape transform sanity check failed")

    texts = dev["text"].to_numpy()
    true = dev["model"].to_numpy()
    base_pred = np.empty(len(dev), dtype=object)
    shape_pred = np.empty(len(dev), dtype=object)
    shape_scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    baseline_score_max_abs_difference = 0.0
    saved_base_scores = original_scores.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float32)
    for fold in range(1, N_FOLDS + 1):
        fit_idx, val_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(dev.loc[fit_idx, "prompt_id"]) & set(dev.loc[val_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompts")
        if set(true[fit_idx]) != set(LABELS) or set(true[val_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} is missing a class")

        base_model = make_pipeline().fit(texts[fit_idx], true[fit_idx])
        base_pred[val_idx] = base_model.predict(texts[val_idx])
        base_margin = ordered_scores(base_model, texts[val_idx])
        baseline_score_max_abs_difference = max(
            baseline_score_max_abs_difference,
            float(np.max(np.abs(base_margin - saved_base_scores[val_idx]))),
        )
        if not np.array_equal(base_pred[val_idx], expected.loc[val_idx, "pred_model"].to_numpy()):
            raise ValueError(f"Original E0c predictions failed to reproduce in fold {fold}")

        shape_model = make_shape_pipeline().fit(texts[fit_idx], true[fit_idx])
        shape_pred[val_idx] = shape_model.predict(texts[val_idx])
        shape_scores[val_idx] = ordered_scores(shape_model, texts[val_idx])
        print(f"Fold {fold}/{N_FOLDS}: E0c={balanced_accuracy_score(true[val_idx], base_pred[val_idx]):.6f}, "
              f"shape={balanced_accuracy_score(true[val_idx], shape_pred[val_idx]):.6f}", flush=True)

    if not np.isfinite(shape_scores).all():
        raise ValueError("Incomplete shape OOF scores")
    if baseline_score_max_abs_difference > 1e-5:
        raise ValueError(f"Reproduced E0c margins differ from saved scores: {baseline_score_max_abs_difference}")
    base_ba = float(balanced_accuracy_score(true, base_pred))
    if not np.isclose(base_ba, 0.34259259259259256, atol=1e-12):
        raise ValueError("Original E0c pooled BA did not reproduce")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = original_scores.loc[:, list(KEY_COLUMNS)].copy()
    oof["pred_model"] = shape_pred
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    scores = original_scores.loc[:, list(KEY_COLUMNS)].copy()
    for index, label in enumerate(LABELS):
        scores[f"score_{label}"] = shape_scores[:, index]
    scores.to_csv(args.output_dir / "decision_scores.csv", index=False)
    metadata = {
        "experiment": "e20_shape_ablation",
        "stage": "fit",
        "train_sha256": sha256(args.train),
        "original_e0c_scores_sha256": sha256(args.e0c_scores),
        "original_e0c_predictions_sha256": sha256(args.e0c_predictions),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "baseline_predictions_exactly_reproduced": True,
        "baseline_score_max_abs_difference": baseline_score_max_abs_difference,
        "baseline_oof_ba": base_ba,
        "shape_oof_ba": float(balanced_accuracy_score(true, shape_pred)),
        "n_rows": len(dev),
        "n_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": N_FOLDS,
        "feature_rule": "re.sub(r'[^\\W\\d_]+', lambda m: 'X' if m[0].isupper() else 'x', text)",
        "shape_tfidf": {
            "analyzer": "char", "ngram_range": [2, 6], "lowercase": False,
            "sublinear_tf": True, "min_df": 2, "max_features": 60000,
            "dtype": "float32", "feature_union_weight": 0.5,
        },
        "original_e0c_blocks_unchanged": True,
        "classifier_unchanged": "LinearSVC(C=1, class_weight='balanced', dual=True, max_iter=10000, random_state=42)",
    }
    (args.output_dir / "fit_metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"baseline_oof_ba": base_ba, "shape_oof_ba": metadata["shape_oof_ba"],
                      "baseline_score_max_abs_difference": baseline_score_max_abs_difference}, indent=2), flush=True)


def analyze(args: argparse.Namespace) -> None:
    # Reuse the E16 configuration/score verification and exact row alignment.
    from analyze_modernbert_seeds import validate_keys, validate_run, score_columns
    from run_score_normalization import paired_prompt_bootstrap, row_standardize

    fit_meta = json.loads((args.fit_dir / "fit_metrics.json").read_text(encoding="utf-8"))
    if not fit_meta["baseline_predictions_exactly_reproduced"]:
        raise ValueError("E0c baseline reproduction was not verified")
    original = load_scores(args.e0c_scores)
    shape = load_scores(args.fit_dir / "decision_scores.csv")
    roberta = load_scores(args.roberta_scores)
    reference_config = json.loads((args.seed42_dir / "run_config.json").read_text(encoding="utf-8"))
    seed_dirs = {42: args.seed42_dir, 17: args.seed17_dir, 73: args.seed73_dir}
    modern = {seed: validate_run(seed, seed_dirs[seed], reference_config)
              for seed in (42, 17, 73)}
    if len(original) != 2160 or original["prompt_id"].nunique() != 120:
        raise ValueError("Original E0c OOF does not have 2,160 rows and 120 prompts")
    for name, data in [("shape", shape), ("RoBERTa", roberta),
                       *[(f"ModernBERT seed {seed}", data) for seed, data in modern.items()]]:
        validate_keys(original, data, name)
    saved_e16 = pd.read_csv(args.e16_oof, dtype=str, keep_default_na=False)
    validate_keys(original, saved_e16, "Saved E16")
    saved_shape = pd.read_csv(args.fit_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    validate_keys(original, saved_shape, "Saved E20 shape")

    labels = np.asarray(LABELS)
    true, prompts = original["true_model"].to_numpy(), original["prompt_id"].to_numpy()
    base_scores, new_scores = score_columns(original), score_columns(shape)
    base_pred, new_pred = labels[base_scores.argmax(axis=1)], labels[new_scores.argmax(axis=1)]
    if not np.array_equal(new_pred, saved_shape["pred_model"].to_numpy()):
        raise ValueError("Shape margin argmax differs from saved shape predictions")
    if not np.array_equal(base_pred, pd.read_csv(args.e0c_predictions, dtype=str)["pred_model"].to_numpy()):
        raise ValueError("Original E0c margin argmax differs from saved predictions")
    mean_modern = np.mean(np.stack([row_standardize(score_columns(modern[seed]))
                                    for seed in (42, 17, 73)]), axis=0)
    zroberta = row_standardize(score_columns(roberta))
    old_e16_pred = labels[(row_standardize(base_scores) + zroberta + mean_modern).argmax(axis=1)]
    new_e16_pred = labels[(row_standardize(new_scores) + zroberta + mean_modern).argmax(axis=1)]
    if not np.array_equal(old_e16_pred, saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("Original E16 fusion failed to reproduce saved OOF predictions")
    if not np.isclose(balanced_accuracy_score(true, old_e16_pred), 0.40925925925925927, atol=1e-12):
        raise ValueError("Original E16 BA failed to reproduce")

    predictions = {"e0c": base_pred, "e20_shape": new_pred,
                   "e16": old_e16_pred, "e16_with_shape": new_e16_pred}
    fold_rows = []
    for fold in range(1, 6):
        mask = original["fold"].eq(str(fold)).to_numpy()
        fold_rows.append({"fold": fold, **{
            name: float(balanced_accuracy_score(true[mask], pred[mask]))
            for name, pred in predictions.items()
        }})
    fold_metrics = pd.DataFrame(fold_rows)
    class_rows = []
    for label in LABELS:
        mask = true == label
        class_rows.append({"model": label, "support": int(mask.sum()), **{
            name: float((pred[mask] == label).mean()) for name, pred in predictions.items()
        }})
    class_metrics = pd.DataFrame(class_rows)
    if not (class_metrics["support"] == 120).all():
        raise ValueError("Expected 120 examples in each class")
    class_metrics["shape_minus_e0c"] = class_metrics["e20_shape"] - class_metrics["e0c"]
    class_metrics["fusion_delta"] = class_metrics["e16_with_shape"] - class_metrics["e16"]
    fold_metrics["shape_minus_e0c"] = fold_metrics["e20_shape"] - fold_metrics["e0c"]
    fold_metrics["fusion_delta"] = fold_metrics["e16_with_shape"] - fold_metrics["e16"]
    oof = original.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in predictions.items():
        oof[f"pred_{name}"] = pred
    metrics = {
        "experiment": "e20_shape_ablation",
        "stage": "analysis",
        "fit_metrics_sha256": sha256(args.fit_dir / "fit_metrics.json"),
        "shape_scores_sha256": sha256(args.fit_dir / "decision_scores.csv"),
        "original_e0c_scores_sha256": sha256(args.e0c_scores),
        "roberta_scores_sha256": sha256(args.roberta_scores),
        "modern_seed_score_sha256": {str(seed): sha256(seed_dirs[seed] / "oof_predictions.csv")
                                     for seed in (42, 17, 73)},
        "e16_saved_oof_sha256": sha256(args.e16_oof),
        "saved_e16_predictions_exactly_reproduced": True,
        "score": {name: float(balanced_accuracy_score(true, pred))
                  for name, pred in predictions.items()},
        "paired_prompt_bootstrap": {
            "shape_minus_e0c": paired_prompt_bootstrap(true, prompts, base_pred, new_pred),
            "fusion_delta": paired_prompt_bootstrap(true, prompts, old_e16_pred, new_e16_pred),
        },
        "fusion_wrong_to_right": int(((old_e16_pred != true) & (new_e16_pred == true)).sum()),
        "fusion_right_to_wrong": int(((old_e16_pred == true) & (new_e16_pred != true)).sum()),
        "fusion_fold_non_decrease": int((fold_metrics["fusion_delta"] >= -1e-12).sum()),
        "worst_fusion_class_recall_delta": float(class_metrics["fusion_delta"].min()),
        "fixed_fusion_rule": "row-z E0c or E20 + row-z RoBERTa + mean of three row-z ModernBERT seeds",
        "weights_tuned": False,
        "holdout_evaluated": False,
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--train", type=Path, default=Path("data/train.csv"))
    fitting.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    fitting.add_argument("--e0c-predictions", type=Path, default=PREDICTION_PATH)
    fitting.add_argument("--e0c-scores", type=Path, default=SCORE_PATH)
    fitting.add_argument("--output-dir", type=Path, default=Path("outputs/e20_shape_fit"))
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--fit-dir", type=Path, default=Path("outputs/e20_shape_fit"))
    analysis.add_argument("--e0c-predictions", type=Path, default=PREDICTION_PATH)
    analysis.add_argument("--e0c-scores", type=Path, default=SCORE_PATH)
    analysis.add_argument("--roberta-scores", type=Path,
                          default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    analysis.add_argument("--seed42-dir", type=Path, default=SEED_DIRS[42])
    analysis.add_argument("--seed17-dir", type=Path, default=SEED_DIRS[17])
    analysis.add_argument("--seed73-dir", type=Path, default=SEED_DIRS[73])
    analysis.add_argument("--e16-oof", type=Path, default=E16_PATH)
    analysis.add_argument("--output-dir", type=Path, default=Path("outputs/e20_shape_analysis"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
