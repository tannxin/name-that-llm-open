#!/usr/bin/env python3
"""E33: pre-registered four-view sparse CPU probe on official development folds.

Read docs/e33_four_view_sparse_preregistration.md before interpreting scores.
The script never reads the reserved holdout or test data.
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from run_combined import make_pipeline as make_e0c_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
TOKEN_RE = re.compile(TOKEN_PATTERN)
WORD_ONLY_RE = re.compile(r"(?u)^\w+$")
SHAPE_RE = re.compile(r"[^\W\d_]+")

# Frozen before any E33 fold was fitted. This is our compact approximation of
# the function-word view, not a copy of the public notebook's vocabulary.
FUNCTION_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not now
of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves also either
neither per via within without yet although among around beside despite except
inside outside since toward towards upon whereas whether whose may might must
shall ought cannot did didst isnt wasnt werent dont doesnt didnt couldnt wouldnt
shouldnt wont cant isnt arent hasn't haven't hadn't doesn't don't didn't
""".split())

E0_DIR = Path("outputs/e0_tfidf_svm")
E0C_SCORE_PATH = Path("outputs/e0c_tfidf_plus_style/decision_scores.csv")
E0C_PRED_PATH = Path("outputs/e0c_tfidf_plus_style/oof_predictions.csv")
E16_PRED_PATH = Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv")
ROBERTA_SCORE_PATH = Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv")
MODERN_SCORE_PATHS = {
    42: Path("outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv"),
    17: Path("outputs/e16_modernbert_seed17_eager/oof_predictions.csv"),
    73: Path("outputs/e16_modernbert_seed73_eager/oof_predictions.csv"),
}


def shape_texts(texts: np.ndarray) -> list[str]:
    return [SHAPE_RE.sub(lambda match: "X" if match[0].isupper() else "x", str(text))
            for text in texts]


def function_texts(texts: np.ndarray) -> list[str]:
    transformed = []
    for text in texts:
        tokens = TOKEN_RE.findall(str(text))
        transformed.append(" ".join(
            token if (WORD_ONLY_RE.fullmatch(token) is None
                      or token.casefold() in FUNCTION_WORDS)
            else "CONTENT" for token in tokens
        ))
    return transformed


def make_four_view_pipeline() -> Pipeline:
    def tfidf(analyzer: str, ngram: tuple[int, int], maximum: int,
              token_pattern: str | None = None) -> TfidfVectorizer:
        kwargs = dict(
            analyzer=analyzer, ngram_range=ngram, min_df=2,
            max_features=maximum, lowercase=False, sublinear_tf=True,
            dtype=np.float32,
        )
        if token_pattern is not None:
            kwargs["token_pattern"] = token_pattern
        return TfidfVectorizer(**kwargs)

    features = FeatureUnion([
        ("char", tfidf("char", (3, 5), 250_000)),
        ("word_punct", tfidf("word", (1, 3), 180_000, TOKEN_PATTERN)),
        ("shape", Pipeline([
            ("extract", FunctionTransformer(shape_texts, validate=False)),
            ("tfidf", tfidf("char", (2, 6), 60_000)),
        ])),
        ("function", Pipeline([
            ("extract", FunctionTransformer(function_texts, validate=False)),
            ("tfidf", tfidf("word", (1, 3), 50_000, TOKEN_PATTERN)),
        ])),
    ], transformer_weights={"char": 1.0, "word_punct": 0.7,
                            "shape": 0.5, "function": 0.5})
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(
            C=0.5, class_weight="balanced", dual=True, max_iter=5000,
            random_state=42,
        )),
    ])


def ordered_scores(model: Pipeline, texts: np.ndarray) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_
    positions = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    return model.decision_function(texts)[:, positions].astype(np.float32)


def validate_keys(reference: pd.DataFrame, other: pd.DataFrame, source: str) -> None:
    for key in KEY_COLUMNS:
        if key not in other or not reference[key].equals(other[key]):
            raise ValueError(f"{source} differs from E0c on {key}")


def load_e16_streams(e0c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    roberta = load_scores(ROBERTA_SCORE_PATH)
    validate_keys(e0c, roberta, "RoBERTa")
    modern = []
    for seed in (42, 17, 73):
        frame = load_scores(MODERN_SCORE_PATHS[seed])
        validate_keys(e0c, frame, f"ModernBERT seed {seed}")
        modern.append(row_standardize(frame.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)))
    zroberta = row_standardize(roberta.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
    zmodern = np.mean(np.stack(modern), axis=0)
    original_sparse = row_standardize(e0c.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
    original_pred = np.asarray(LABELS)[(original_sparse + zroberta + zmodern).argmax(axis=1)]
    saved = pd.read_csv(E16_PRED_PATH, dtype=str, keep_default_na=False)
    validate_keys(e0c, saved, "Saved E16")
    if not np.array_equal(original_pred, saved["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("Original E16 rule does not reproduce saved E16 predictions")
    return zroberta + zmodern, original_pred


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is nonempty: {output_dir}")
    if not Path("docs/e33_four_view_sparse_preregistration.md").is_file():
        raise ValueError("E33 preregistration document is missing")
    if sha256(train_path) != TRAIN_SHA256:
        raise ValueError("Official train.csv differs from the locked dataset")
    if shape_texts(np.asarray(["Hello WORLD x3 _keep.\nAa"])) != ["x X x3 _x.\nx"]:
        raise ValueError("Shape transform sanity check failed")
    if function_texts(np.asarray(["The model, and 42!"], dtype=object)) != ["The CONTENT , and CONTENT !"]:
        raise ValueError("Function transform sanity check failed")

    all_train = load_and_validate(train_path)
    dev, _, _, folds, _ = load_e0_split(all_train, E0_DIR, train_path)
    saved_e0c = load_scores(E0C_SCORE_PATH)
    saved_e0c_pred = pd.read_csv(E0C_PRED_PATH, dtype=str, keep_default_na=False)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Expected 2,160 development rows and 120 prompts")
    for source, frame in (("Saved E0c predictions", saved_e0c_pred),):
        validate_keys(saved_e0c, frame, source)
    if not np.array_equal(dev["id"].to_numpy(), saved_e0c["id"].to_numpy()):
        raise ValueError("Development data rows do not align with saved E0c")
    e16_other_streams, e16_pred = load_e16_streams(saved_e0c)

    texts = dev["text"].to_numpy()
    truth = dev["model"].to_numpy()
    base_pred = np.empty(len(dev), dtype=object)
    four_pred = np.empty(len(dev), dtype=object)
    four_scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    saved_base_scores = saved_e0c.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float32)
    largest_base_score_difference = 0.0

    # Complete the entire same-environment baseline check before any E33 fit.
    for fold in range(1, N_FOLDS + 1):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(dev.loc[fit_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} overlaps prompt IDs")
        if set(truth[fit_idx]) != set(LABELS) or set(truth[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks a class")
        base = make_e0c_pipeline().fit(texts[fit_idx], truth[fit_idx])
        base_pred[valid_idx] = base.predict(texts[valid_idx])
        base_scores = ordered_scores(base, texts[valid_idx])
        largest_base_score_difference = max(
            largest_base_score_difference,
            float(np.max(np.abs(base_scores - saved_base_scores[valid_idx]))),
        )
        baseline_mismatch = int(np.sum(base_pred[valid_idx] !=
                                       saved_e0c_pred.loc[valid_idx, "pred_model"].to_numpy()))
        print(f"E0c local fold {fold}: mismatches vs saved={baseline_mismatch}, "
              f"BA={balanced_accuracy_score(truth[valid_idx], base_pred[valid_idx]):.6f}",
              flush=True)
        del base

    mismatch_count = int(np.sum(base_pred != saved_e0c_pred["pred_model"].to_numpy()))
    print(f"E0c local full baseline: mismatches={mismatch_count}, "
          f"max margin difference={largest_base_score_difference:.8f}", flush=True)

    for fold in range(1, N_FOLDS + 1):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = make_four_view_pipeline().fit(texts[fit_idx], truth[fit_idx])
        four_pred[valid_idx] = model.predict(texts[valid_idx])
        four_scores[valid_idx] = ordered_scores(model, texts[valid_idx])
        print(f"E33 fold {fold}: BA={balanced_accuracy_score(truth[valid_idx], four_pred[valid_idx]):.6f}",
              flush=True)
        del model

    if not np.isfinite(four_scores).all():
        raise ValueError("E33 OOF scores are incomplete")
    four_z = row_standardize(four_scores.astype(np.float64))
    e16_replace_pred = np.asarray(LABELS)[(four_z + e16_other_streams).argmax(axis=1)]
    saved_e0c_pred_values = saved_e0c_pred["pred_model"].to_numpy()
    baseline_reproduced = mismatch_count == 0 and largest_base_score_difference <= 1e-5
    candidates = {
        "saved_e0c": saved_e0c_pred_values,
        "local_e0c": base_pred,
        "e33_four_view": four_pred,
        "saved_e16": e16_pred,
        "e16_e33_sparse_replace": e16_replace_pred,
    }
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        mask = folds == fold
        fold_rows.append({
            "fold": fold,
            "n_rows": int(mask.sum()),
            "n_prompts": int(dev.loc[mask, "prompt_id"].nunique()),
            **{name: float(balanced_accuracy_score(truth[mask], pred[mask]))
               for name, pred in candidates.items()},
        })
    class_rows = []
    for label in LABELS:
        mask = truth == label
        class_rows.append({
            "model": label, "support": int(mask.sum()),
            **{name: float((pred[mask] == label).mean()) for name, pred in candidates.items()},
        })
    fold_metrics = pd.DataFrame(fold_rows)
    class_metrics = pd.DataFrame(class_rows)
    class_delta = (class_metrics["e16_e33_sparse_replace"] - class_metrics["saved_e16"])
    fusion_delta = float(balanced_accuracy_score(truth, e16_replace_pred) -
                         balanced_accuracy_score(truth, e16_pred))
    fold_non_decrease = int((fold_metrics["e16_e33_sparse_replace"] >=
                             fold_metrics["saved_e16"] - 1e-12).sum())
    gate = {
        "local_e0c_matches_saved_predictions_and_margins": baseline_reproduced,
        "fusion_delta_at_least_0_01": fusion_delta >= 0.01 - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": fold_non_decrease >= 4,
        "no_class_recall_drop_at_least_0_05": float(class_delta.min()) > -0.05 + 1e-12,
    }
    metrics = {
        "experiment": "e33_four_view_sparse",
        "status": "confirmatory_local" if baseline_reproduced else "exploratory_environment_mismatch",
        "gate_passed": all(gate.values()),
        "gate": gate,
        "train_sha256": sha256(train_path),
        "preregistration_sha256": sha256(Path("docs/e33_four_view_sparse_preregistration.md")),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "n_development_rows": len(dev),
        "n_development_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": N_FOLDS,
        "holdout_scored": False,
        "test_scored": False,
        "baseline_prediction_mismatch_rows": mismatch_count,
        "baseline_score_max_abs_difference": largest_base_score_difference,
        "pooled_balanced_accuracy": {
            name: float(balanced_accuracy_score(truth, pred))
            for name, pred in candidates.items()
        },
        "fusion_delta_vs_e16": fusion_delta,
        "fusion_fold_non_decrease": fold_non_decrease,
        "worst_fusion_class_recall_delta": float(class_delta.min()),
        "worst_fusion_class": str(class_metrics.loc[class_delta.idxmin(), "model"]),
        "paired_prompt_bootstrap_vs_e16": paired_prompt_bootstrap(
            truth, dev["prompt_id"].to_numpy(), e16_pred, e16_replace_pred),
    }
    output_dir.mkdir(parents=True)
    keys = saved_e0c.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in candidates.items():
        keys[f"pred_{name}"] = pred
    keys.to_csv(output_dir / "oof_predictions.csv", index=False)
    scores = saved_e0c.loc[:, list(KEY_COLUMNS)].copy()
    for index, label in enumerate(LABELS):
        scores[f"score_{label}"] = four_scores[:, index]
    scores.to_csv(output_dir / "decision_scores.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e33_four_view_sparse"))
    args = parser.parse_args()
    run(args.train, args.output_dir)


if __name__ == "__main__":
    main()
