#!/usr/bin/env python3
"""E0b: text-format and stylometric features on E0's fixed development folds.

The E0 manifest and out-of-fold predictions are required inputs. This script
never selects, transforms, fits on, or scores the reserved holdout rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256


WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
NUMBERED_LINE_PATTERN = re.compile(r"\s*\d+[.)]\s+")
HEADING_LINE_PATTERN = re.compile(r"\s*#{1,6}\s+")
LIST_LINE_PATTERN = re.compile(r"\s*[-*+]\s+")
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)

FEATURE_NAMES = (
    "log_chars", "log_words", "log_lines", "log_paragraphs",
    "mean_word_length", "std_word_length", "mean_line_length",
    "std_line_length", "max_line_length", "chars_per_word",
    "words_per_sentence", "unique_word_ratio",
    "letter_fraction", "digit_fraction", "whitespace_fraction",
    "punctuation_fraction", "symbol_fraction", "uppercase_letter_fraction",
    "non_ascii_fraction", "newline_per_1000", "tab_per_1000",
    "period_per_1000", "comma_per_1000", "semicolon_per_1000",
    "colon_per_1000", "question_per_1000", "exclamation_per_1000",
    "hyphen_per_1000", "unicode_dash_per_1000", "slash_per_1000",
    "pipe_per_1000", "parenthesis_per_1000", "bracket_per_1000",
    "brace_per_1000", "double_quote_per_1000", "single_quote_per_1000",
    "backtick_per_1000", "asterisk_per_1000", "hash_per_1000",
    "bullet_per_1000", "ellipsis_per_1000", "ampersand_per_1000",
    "tilde_per_1000", "equals_per_1000", "angle_per_1000",
    "underscore_per_1000", "url_per_1000", "sentence_end_per_1000",
    "blank_line_fraction", "short_line_fraction", "indented_line_fraction",
    "heading_line_fraction", "bullet_line_fraction",
    "numbered_line_fraction", "blockquote_line_fraction",
    "code_fence_line_fraction", "trailing_space_line_fraction",
)


def extract_one(text: str) -> np.ndarray:
    """Return fixed style features without learning a word vocabulary."""
    chars = len(text)
    denom = max(chars, 1)
    per_1000 = 1000 / denom
    categories = [unicodedata.category(char) for char in text]
    letters = sum(category.startswith("L") for category in categories)
    words = WORD_PATTERN.findall(text)
    word_lengths = [len(word) for word in words]
    n_words = len(words)
    lines = text.splitlines() or [text]
    n_lines = len(lines)
    line_lengths = [len(line) for line in lines]
    paragraphs = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    sentence_ends = sum(char in ".!?。！？" for char in text)
    sentence_count = max(sentence_ends, 1)

    def count_any(charset: str) -> int:
        return sum(char in charset for char in text)

    def line_fraction(predicate) -> float:
        return sum(bool(predicate(line)) for line in lines) / n_lines

    features = {
        "log_chars": np.log1p(chars),
        "log_words": np.log1p(n_words),
        "log_lines": np.log1p(n_lines),
        "log_paragraphs": np.log1p(len(paragraphs)),
        "mean_word_length": np.mean(word_lengths) if word_lengths else 0,
        "std_word_length": np.std(word_lengths) if word_lengths else 0,
        "mean_line_length": np.mean(line_lengths),
        "std_line_length": np.std(line_lengths),
        "max_line_length": max(line_lengths),
        "chars_per_word": chars / max(n_words, 1),
        "words_per_sentence": n_words / sentence_count,
        "unique_word_ratio": len({word.casefold() for word in words}) / max(n_words, 1),
        "letter_fraction": letters / denom,
        "digit_fraction": sum(category == "Nd" for category in categories) / denom,
        "whitespace_fraction": sum(char.isspace() for char in text) / denom,
        "punctuation_fraction": sum(category.startswith("P") for category in categories) / denom,
        "symbol_fraction": sum(category.startswith("S") for category in categories) / denom,
        "uppercase_letter_fraction": sum(char.isupper() for char in text) / max(letters, 1),
        "non_ascii_fraction": sum(ord(char) > 127 for char in text) / denom,
        "newline_per_1000": text.count("\n") * per_1000,
        "tab_per_1000": text.count("\t") * per_1000,
        "period_per_1000": count_any(".。") * per_1000,
        "comma_per_1000": count_any(",，、") * per_1000,
        "semicolon_per_1000": count_any(";；") * per_1000,
        "colon_per_1000": count_any(":：") * per_1000,
        "question_per_1000": count_any("?？") * per_1000,
        "exclamation_per_1000": count_any("!！") * per_1000,
        "hyphen_per_1000": text.count("-") * per_1000,
        "unicode_dash_per_1000": count_any("–—−") * per_1000,
        "slash_per_1000": count_any("/\\") * per_1000,
        "pipe_per_1000": text.count("|") * per_1000,
        "parenthesis_per_1000": count_any("()（）") * per_1000,
        "bracket_per_1000": count_any("[]【】") * per_1000,
        "brace_per_1000": count_any("{}") * per_1000,
        "double_quote_per_1000": count_any('"“”') * per_1000,
        "single_quote_per_1000": count_any("'‘’") * per_1000,
        "backtick_per_1000": text.count("`") * per_1000,
        "asterisk_per_1000": text.count("*") * per_1000,
        "hash_per_1000": text.count("#") * per_1000,
        "bullet_per_1000": count_any("•·") * per_1000,
        "ellipsis_per_1000": count_any("…") * per_1000,
        "ampersand_per_1000": text.count("&") * per_1000,
        "tilde_per_1000": text.count("~") * per_1000,
        "equals_per_1000": text.count("=") * per_1000,
        "angle_per_1000": count_any("<>") * per_1000,
        "underscore_per_1000": text.count("_") * per_1000,
        "url_per_1000": len(URL_PATTERN.findall(text)) * per_1000,
        "sentence_end_per_1000": sentence_ends * per_1000,
        "blank_line_fraction": line_fraction(lambda line: not line.strip()),
        "short_line_fraction": line_fraction(lambda line: len(line.strip()) <= 40),
        "indented_line_fraction": line_fraction(lambda line: line.startswith((" ", "\t"))),
        "heading_line_fraction": line_fraction(HEADING_LINE_PATTERN.match),
        "bullet_line_fraction": line_fraction(LIST_LINE_PATTERN.match),
        "numbered_line_fraction": line_fraction(NUMBERED_LINE_PATTERN.match),
        "blockquote_line_fraction": line_fraction(lambda line: line.lstrip().startswith(">")),
        "code_fence_line_fraction": line_fraction(lambda line: line.lstrip().startswith("```")),
        "trailing_space_line_fraction": line_fraction(lambda line: line.endswith((" ", "\t"))),
    }
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_features(texts: pd.Series) -> np.ndarray:
    matrix = np.stack([extract_one(text) for text in texts])
    if not np.isfinite(matrix).all():
        raise ValueError("Style features contain non-finite values")
    return matrix


def load_e0_split(frame: pd.DataFrame, e0_dir: Path, train_path: Path):
    metrics = json.loads((e0_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["train_sha256"] != sha256(train_path):
        raise ValueError("Training file hash differs from E0")
    manifest = pd.read_csv(e0_dir / "split_manifest.csv", dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("E0 manifest has unexpected columns")
    if manifest["prompt_id"].duplicated().any() or set(manifest["prompt_id"]) != set(frame["prompt_id"]):
        raise ValueError("E0 manifest does not cover all unique prompt IDs exactly once")
    if not manifest["split"].isin(["development", "holdout"]).all():
        raise ValueError("E0 manifest contains an unknown split")
    if manifest["split"].eq("holdout").sum() != metrics["n_prompts_holdout"]:
        raise ValueError("E0 manifest holdout count differs from saved metrics")

    development_ids = set(manifest.loc[manifest["split"].eq("development"), "prompt_id"])
    development = frame.loc[frame["prompt_id"].isin(development_ids)].reset_index(drop=True)
    baseline = pd.read_csv(e0_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    required = {"id", "prompt_id", "true_model", "pred_model", "fold"}
    if not required.issubset(baseline.columns):
        raise ValueError("E0 out-of-fold file lacks required columns")
    expected = development.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if not baseline.loc[:, ["id", "prompt_id", "true_model"]].equals(expected):
        raise ValueError("E0 out-of-fold rows do not match development rows in input order")
    folds = pd.to_numeric(baseline["fold"], errors="raise").to_numpy()
    if not np.isin(folds, np.arange(1, N_FOLDS + 1)).all():
        raise ValueError("E0 fold numbers are outside 1..5")
    for _, values in development.assign(fold=folds).groupby("prompt_id")["fold"]:
        if values.nunique() != 1:
            raise ValueError("A prompt appears in more than one E0 fold")
    return development, manifest, baseline, folds, metrics


def make_classifier() -> sklearn.pipeline.Pipeline:
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=10_000, random_state=42),
    )


def run(train_path: Path, e0_dir: Path, output_dir: Path) -> None:
    frame = load_and_validate(train_path)
    development, manifest, baseline, folds, e0_metrics = load_e0_split(frame, e0_dir, train_path)
    if output_dir.resolve() == e0_dir.resolve():
        raise ValueError("Style output directory must differ from E0 directory")
    labels = development["model"].to_numpy()
    matrix = extract_features(development["text"])
    predicted = np.empty(len(development), dtype=object)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks one or more official classes")
        if set(development.loc[fit_idx, "prompt_id"]) & set(development.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompt groups")
        model = make_classifier()
        model.fit(matrix[fit_idx], labels[fit_idx])
        predicted[valid_idx] = model.predict(matrix[valid_idx])
        style_score = float(balanced_accuracy_score(labels[valid_idx], predicted[valid_idx]))
        e0_score = float(balanced_accuracy_score(labels[valid_idx], baseline.loc[valid_idx, "pred_model"]))
        fold_rows.append({
            "fold": fold,
            "train_rows": len(fit_idx),
            "validation_rows": len(valid_idx),
            "train_prompts": development.loc[fit_idx, "prompt_id"].nunique(),
            "validation_prompts": development.loc[valid_idx, "prompt_id"].nunique(),
            "style_balanced_accuracy": style_score,
            "e0_balanced_accuracy": e0_score,
            "difference_vs_e0": style_score - e0_score,
        })
        print(f"Fold {fold}/{N_FOLDS}: style={style_score:.6f}, E0={e0_score:.6f}", flush=True)

    if any(value is None for value in predicted):
        raise RuntimeError("Style OOF predictions are incomplete")
    style_score = float(balanced_accuracy_score(labels, predicted))
    e0_score = float(balanced_accuracy_score(labels, baseline["pred_model"]))
    if not np.isclose(e0_score, e0_metrics["oof_balanced_accuracy"]):
        raise ValueError("Recomputed E0 score differs from its metrics file")

    confusion = confusion_matrix(labels, predicted, labels=LABELS)
    e0_confusion = confusion_matrix(labels, baseline["pred_model"], labels=LABELS)
    class_metrics = pd.DataFrame({
        "model": LABELS,
        "support": confusion.sum(axis=1),
        "style_correct": confusion.diagonal(),
        "style_recall": confusion.diagonal() / confusion.sum(axis=1),
        "e0_correct": e0_confusion.diagonal(),
        "e0_recall": e0_confusion.diagonal() / e0_confusion.sum(axis=1),
    })
    class_metrics["recall_difference_vs_e0"] = class_metrics["style_recall"] - class_metrics["e0_recall"]

    # Fit only on the 120 development prompts for later inference.
    final_model = make_classifier()
    final_model.fit(matrix, labels)
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
        "experiment": "e0b_style_only",
        "metric": "balanced_accuracy",
        "oof_balanced_accuracy": style_score,
        "e0_oof_balanced_accuracy": e0_score,
        "difference_vs_e0": style_score - e0_score,
        "mean_fold_balanced_accuracy": float(np.mean([row["style_balanced_accuracy"] for row in fold_rows])),
        "n_rows_total": len(frame),
        "n_rows_development": len(development),
        "n_prompts_development": development["prompt_id"].nunique(),
        "n_rows_holdout": len(frame) - len(development),
        "n_prompts_holdout": int(manifest["split"].eq("holdout").sum()),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "n_folds": N_FOLDS,
        "fold_metrics": fold_rows,
        "train_sha256": sha256(train_path),
        "e0_metrics_path": str((e0_dir / "metrics.json").resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "classifier": "StandardScaler + LinearSVC(C=1.0, class_weight='balanced')",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"model": final_model, "feature_names": FEATURE_NAMES}, output_dir / "dev_model.joblib", compress=3)
    print(f"OOF balanced_accuracy: style={style_score:.6f}, E0={e0_score:.6f}, difference={style_score - e0_score:+.6f}")
    print(f"Artifacts: {output_dir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e0b_style_only"))
    args = parser.parse_args()
    try:
        run(args.train, args.e0_dir, args.output_dir)
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
