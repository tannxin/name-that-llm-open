#!/usr/bin/env python3
"""E0: prompt-grouped TF-IDF + linear SVM baseline for Name That LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# Official competition spelling and order:
# https://www.kaggle.com/competitions/name-that-llm/overview/description
LABELS = (
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
)
REQUIRED_COLUMNS = ("id", "prompt_id", "text", "model")
N_FOLDS = 5
RANDOM_STATE = 42
DEFAULT_HOLDOUT_FRACTION = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.csv"),
        help="Official competition train.csv (default: data/train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/e0_tfidf_svm"),
        help="Directory for OOF results and fitted model",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Fraction of prompt groups reserved for later final evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Seed for the fixed prompt holdout and LinearSVC (default: 42)",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Training file not found: {path}")

    # Keep IDs and model names exactly as provided. No text or label normalization.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(
            f"Expected exactly {list(REQUIRED_COLUMNS)}; "
            f"missing={missing_columns}, extra={extra_columns}"
        )
    if frame.empty:
        raise ValueError("Training CSV is empty")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        blank = frame[column].isna() | frame[column].str.strip().eq("")
        if blank.any():
            raise ValueError(f"Column {column!r} has {int(blank.sum())} blank rows")

    for column in ("id", "prompt_id", "model"):
        padded = frame[column].ne(frame[column].str.strip())
        if padded.any():
            raise ValueError(f"Column {column!r} has leading or trailing whitespace")

    duplicate_ids = frame["id"].duplicated()
    if duplicate_ids.any():
        raise ValueError(f"Found {int(duplicate_ids.sum())} duplicate id values")
    duplicate_prompt_model = frame.duplicated(["prompt_id", "model"])
    if duplicate_prompt_model.any():
        raise ValueError(
            f"Found {int(duplicate_prompt_model.sum())} repeated "
            "(prompt_id, model) pairs"
        )

    found = set(frame["model"])
    expected = set(LABELS)
    if found != expected:
        raise ValueError(
            "Training labels differ from the 18 official labels. "
            f"Missing={sorted(expected - found)}, unexpected={sorted(found - expected)}"
        )
    n_prompts = frame["prompt_id"].nunique()
    if n_prompts < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} unique prompts; found {n_prompts}")
    return frame


def make_pipeline(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual=True,
                    max_iter=10_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_development_split(
    frame: pd.DataFrame, holdout_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 <= holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be in [0, 1)")
    prompts = np.array(sorted(frame["prompt_id"].unique()))
    n_holdout = round(len(prompts) * holdout_fraction)
    if holdout_fraction > 0 and n_holdout == 0:
        raise ValueError("Holdout fraction is too small to reserve a prompt")
    if len(prompts) - n_holdout < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} development prompts after holdout")

    held_out = set(np.random.default_rng(seed).permutation(prompts)[:n_holdout])
    split_manifest = pd.DataFrame({"prompt_id": prompts})
    split_manifest["split"] = np.where(
        split_manifest["prompt_id"].isin(held_out), "holdout", "development"
    )
    development = frame.loc[~frame["prompt_id"].isin(held_out)].reset_index(drop=True)
    return development, split_manifest


def run(train_path: Path, output_dir: Path, holdout_fraction: float, seed: int) -> None:
    frame = load_and_validate(train_path)
    development, split_manifest = make_development_split(frame, holdout_fraction, seed)
    existing_manifest = output_dir / "split_manifest.csv"
    if existing_manifest.exists():
        previous = pd.read_csv(existing_manifest, dtype=str, keep_default_na=False)
        if not previous.equals(split_manifest):
            raise ValueError(
                f"The saved prompt split in {existing_manifest} differs from this run. "
                "Use the original seed and fraction, or choose a new output directory."
            )

    labels = development["model"].to_numpy()
    groups = development["prompt_id"].to_numpy()
    texts = development["text"].to_numpy()

    oof_predictions = np.empty(len(development), dtype=object)
    fold_for_row = np.zeros(len(development), dtype=np.int8)
    fold_rows: list[dict[str, int | float]] = []
    splitter = GroupKFold(n_splits=N_FOLDS)

    for fold, (fit_idx, valid_idx) in enumerate(
        splitter.split(texts, labels, groups=groups), start=1
    ):
        fit_groups = set(groups[fit_idx])
        valid_groups = set(groups[valid_idx])
        if fit_groups & valid_groups:
            raise RuntimeError(f"Fold {fold} has overlapping prompt_id groups")
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(
                f"Fold {fold} does not contain all 18 labels in both partitions; "
                "balanced accuracy would not be comparable"
            )

        pipeline = make_pipeline(seed)
        pipeline.fit(texts[fit_idx], labels[fit_idx])
        predicted = pipeline.predict(texts[valid_idx])
        oof_predictions[valid_idx] = predicted
        fold_for_row[valid_idx] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(fit_idx)),
                "validation_rows": int(len(valid_idx)),
                "train_prompts": len(fit_groups),
                "validation_prompts": len(valid_groups),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[valid_idx], predicted)
                ),
            }
        )
        print(
            f"Fold {fold}/{N_FOLDS}: balanced_accuracy="
            f"{fold_rows[-1]['balanced_accuracy']:.6f}",
            flush=True,
        )

    if np.any(fold_for_row == 0) or any(pred is None for pred in oof_predictions):
        raise RuntimeError("OOF predictions do not cover every training row")

    oof_score = float(balanced_accuracy_score(labels, oof_predictions))
    confusion = confusion_matrix(labels, oof_predictions, labels=LABELS)
    class_metrics = pd.DataFrame(
        {
            "model": LABELS,
            "support": confusion.sum(axis=1),
            "correct": confusion.diagonal(),
        }
    )
    class_metrics["recall"] = class_metrics["correct"] / class_metrics["support"]

    # The holdout is untouched. This reusable model sees development rows only.
    development_pipeline = make_pipeline(seed)
    development_pipeline.fit(texts, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    oof = development.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["pred_model"] = oof_predictions
    oof["fold"] = fold_for_row
    oof["correct"] = labels == oof_predictions
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(output_dir / "confusion_matrix.csv")
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)

    metrics = {
        "experiment": "e0_tfidf_svm",
        "metric": "balanced_accuracy",
        "oof_balanced_accuracy": oof_score,
        "mean_fold_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in fold_rows])
        ),
        "n_rows_total": len(frame),
        "n_prompts_total": int(frame["prompt_id"].nunique()),
        "n_rows_development": len(development),
        "n_prompts_development": int(development["prompt_id"].nunique()),
        "n_rows_holdout": len(frame) - len(development),
        "n_prompts_holdout": int(split_manifest["split"].eq("holdout").sum()),
        "holdout_fraction": holdout_fraction,
        "n_classes": len(LABELS),
        "n_folds": N_FOLDS,
        "fold_metrics": fold_rows,
        "labels": list(LABELS),
        "train_file": str(train_path.resolve()),
        "train_sha256": sha256(train_path),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "random_state": seed,
        "classifier": "LinearSVC(C=1.0, class_weight='balanced', dual=True, max_iter=10000)",
        "features": {
            "word": "TF-IDF, ngram_range=(1,2), min_df=2, max_features=50000",
            "char": "TF-IDF, ngram_range=(3,5), min_df=2, max_features=100000",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(development_pipeline, output_dir / "dev_model.joblib", compress=3)
    print(f"OOF balanced_accuracy={oof_score:.6f}")
    print(f"Artifacts: {output_dir.resolve()}")


def main() -> int:
    args = parse_args()
    try:
        run(args.train, args.output_dir, args.holdout_fraction, args.seed)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
