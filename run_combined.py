#!/usr/bin/env python3
"""E0c: E0 word/character TF-IDF plus the E0b style feature block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import FEATURE_NAMES, extract_features, load_e0_split


def make_pipeline() -> Pipeline:
    # Each TF-IDF block has unit L2 norm per response. Standardize the style
    # columns using fit-fold rows only, then normalize that block to unit norm.
    # This gives the three blocks comparable scale without tuning on validation.
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=50_000, sublinear_tf=True,
            lowercase=False, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            max_features=100_000, sublinear_tf=True,
            lowercase=False, dtype=np.float32,
        )),
        ("style", Pipeline([
            ("extract", FunctionTransformer(extract_features, validate=False)),
            ("standardize", StandardScaler()),
            ("normalize", Normalizer()),
        ])),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(
            C=1.0, class_weight="balanced", dual=True,
            max_iter=10_000, random_state=42,
        )),
    ])


def run(train_path: Path, e0_dir: Path, e0b_dir: Path, output_dir: Path) -> None:
    frame = load_and_validate(train_path)
    development, manifest, e0, folds, e0_metrics = load_e0_split(frame, e0_dir, train_path)
    if output_dir.resolve() in {e0_dir.resolve(), e0b_dir.resolve()}:
        raise ValueError("Combined output directory must differ from E0 and E0b directories")
    e0b = pd.read_csv(e0b_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for column in ("id", "prompt_id", "true_model", "fold"):
        if column not in e0b or not e0b[column].equals(e0[column]):
            raise ValueError(f"E0b {column} does not match E0 development rows")
    if "pred_model" not in e0b:
        raise ValueError("E0b predictions are missing")

    labels = development["model"].to_numpy()
    texts = development["text"].to_numpy()
    predicted = np.empty(len(development), dtype=object)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks one or more official classes")
        if set(development.loc[fit_idx, "prompt_id"]) & set(development.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompt groups")
        model = make_pipeline()
        model.fit(texts[fit_idx], labels[fit_idx])
        predicted[valid_idx] = model.predict(texts[valid_idx])
        score = float(balanced_accuracy_score(labels[valid_idx], predicted[valid_idx]))
        e0_score = float(balanced_accuracy_score(labels[valid_idx], e0.loc[valid_idx, "pred_model"]))
        e0b_score = float(balanced_accuracy_score(labels[valid_idx], e0b.loc[valid_idx, "pred_model"]))
        fold_rows.append({
            "fold": fold,
            "train_rows": len(fit_idx),
            "validation_rows": len(valid_idx),
            "train_prompts": development.loc[fit_idx, "prompt_id"].nunique(),
            "validation_prompts": development.loc[valid_idx, "prompt_id"].nunique(),
            "combined_balanced_accuracy": score,
            "e0_balanced_accuracy": e0_score,
            "e0b_balanced_accuracy": e0b_score,
            "difference_vs_e0": score - e0_score,
        })
        print(f"Fold {fold}/{N_FOLDS}: combined={score:.6f}, E0={e0_score:.6f}, E0b={e0b_score:.6f}", flush=True)
    if any(value is None for value in predicted):
        raise RuntimeError("Combined OOF predictions are incomplete")

    score = float(balanced_accuracy_score(labels, predicted))
    e0_score = float(balanced_accuracy_score(labels, e0["pred_model"]))
    e0b_score = float(balanced_accuracy_score(labels, e0b["pred_model"]))
    e0b_metrics = json.loads((e0b_dir / "metrics.json").read_text(encoding="utf-8"))
    if not np.isclose(e0_score, e0_metrics["oof_balanced_accuracy"]):
        raise ValueError("Recomputed E0 score differs from saved metrics")
    if not np.isclose(e0b_score, e0b_metrics["oof_balanced_accuracy"]):
        raise ValueError("Recomputed E0b score differs from saved metrics")

    confusion = confusion_matrix(labels, predicted, labels=LABELS)
    e0_confusion = confusion_matrix(labels, e0["pred_model"], labels=LABELS)
    e0b_confusion = confusion_matrix(labels, e0b["pred_model"], labels=LABELS)
    support = confusion.sum(axis=1)
    class_metrics = pd.DataFrame({
        "model": LABELS,
        "support": support,
        "combined_correct": confusion.diagonal(),
        "combined_recall": confusion.diagonal() / support,
        "e0_recall": e0_confusion.diagonal() / support,
        "e0b_recall": e0b_confusion.diagonal() / support,
    })
    class_metrics["recall_difference_vs_e0"] = class_metrics["combined_recall"] - class_metrics["e0_recall"]
    class_metrics["recall_difference_vs_e0b"] = class_metrics["combined_recall"] - class_metrics["e0b_recall"]

    # Reserve the held-out 30 prompts throughout feature fitting and training.
    final_model = make_pipeline()
    final_model.fit(texts, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    oof = development.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["pred_model"] = predicted
    oof["fold"] = folds
    oof["correct"] = labels == predicted
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(output_dir / "confusion_matrix.csv")
    metrics = {
        "experiment": "e0c_tfidf_plus_style",
        "metric": "balanced_accuracy",
        "oof_balanced_accuracy": score,
        "e0_oof_balanced_accuracy": e0_score,
        "e0b_oof_balanced_accuracy": e0b_score,
        "difference_vs_e0": score - e0_score,
        "difference_vs_e0b": score - e0b_score,
        "mean_fold_balanced_accuracy": float(np.mean([row["combined_balanced_accuracy"] for row in fold_rows])),
        "n_rows_total": len(frame),
        "n_rows_development": len(development),
        "n_prompts_development": development["prompt_id"].nunique(),
        "n_rows_holdout": len(frame) - len(development),
        "n_prompts_holdout": int(manifest["split"].eq("holdout").sum()),
        "n_style_features": len(FEATURE_NAMES),
        "n_folds": N_FOLDS,
        "fold_metrics": fold_rows,
        "train_sha256": sha256(train_path),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "feature_blocks": {
            "word": "E0 TF-IDF word 1-2 gram",
            "char": "E0 TF-IDF character 3-5 gram",
            "style": "E0b fixed features, StandardScaler then row L2 normalization",
        },
        "classifier": "LinearSVC(C=1.0, class_weight='balanced', dual=True)",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    joblib.dump(final_model, output_dir / "dev_model.joblib", compress=3)
    print(f"OOF balanced_accuracy: combined={score:.6f}, E0={e0_score:.6f}, E0b={e0b_score:.6f}")
    print(f"Difference vs E0: {score - e0_score:+.6f}")
    print(f"Artifacts: {output_dir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0b-dir", type=Path, default=Path("outputs/e0b_style_only"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    args = parser.parse_args()
    try:
        run(args.train, args.e0_dir, args.e0b_dir, args.output_dir)
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
