#!/usr/bin/env python3
"""Prompt-grouped E0c ablation for exact punctuation fingerprints.

This is an exploratory development experiment. It reuses the E0 manifest and
fold IDs, fits every transform inside each training fold, and never extracts
features from or scores the reserved 30-prompt holdout.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import FEATURE_NAMES, extract_features, load_e0_split


# Specified before looking at validation results. These distinguish characters
# that the original 57-feature block counts together. Rates are per 1,000
# characters, matching the original style features.
PUNCTUATION_NAMES = (
    "ascii_double_quote_per_1000",
    "curly_double_open_per_1000",
    "curly_double_close_per_1000",
    "ascii_single_quote_per_1000",
    "curly_single_open_per_1000",
    "curly_single_close_per_1000",
    "em_dash_per_1000",
    "en_dash_per_1000",
    "minus_sign_per_1000",
    "ascii_three_dot_ellipsis_per_1000",
)

# A fixed, English closed-class vocabulary. Keep this hand-specified rather
# than selecting tokens from development labels or validation predictions.
FUNCTION_WORDS = frozenset("""
    a an the this that these those each every either neither some any no all
    another other such both few many much more most less least enough
    i me my mine myself we us our ours ourselves you your yours yourself
    yourselves he him his himself she her hers herself it its itself they them
    their theirs themselves who whom whose which what whoever whomever
    am is are was were be being been have has had having do does did doing
    can could may might must shall should will would ought
    not nor and or but yet so for if then than because although though while
    whereas unless until since before after once as whether when whenever
    where wherever how however why although even also only just still already
    very too quite rather almost never ever always often sometimes usually
    here there now again above below up down in out on off over under into
    onto through throughout across along around between among beyond behind
    beside besides within without toward towards against about by with from
    to at of upon per via like unlike except despite during amid versus
    yes no perhaps maybe indeed therefore moreover furthermore however
    instead otherwise nonetheless nevertheless thus hence
    it's i'm i've i'll i'd you're you've you'll you'd we're we've we'll we'd
    they're they've they'll they'd he's she's that's there's here's what's
    who's let's don't doesn't didn't can't cannot couldn't won't wouldn't
    shouldn't isn't aren't wasn't weren't hasn't haven't hadn't
""".split())
FUNCTION_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def function_word_sequence(text: str) -> str:
    """Remove open-class words and preserve the order of function words."""
    tokens = FUNCTION_TOKEN_PATTERN.findall(text.lower())
    return " ".join(token.replace("'", "_") for token in tokens if token in FUNCTION_WORDS)


def extract_punctuation_features(texts: pd.Series | np.ndarray) -> np.ndarray:
    """Append exact punctuation rates to the unchanged original style block."""
    original = extract_features(texts)
    extra = np.empty((len(texts), len(PUNCTUATION_NAMES)), dtype=np.float64)
    for row, value in enumerate(texts):
        text = str(value)
        rate = 1000.0 / max(len(text), 1)
        counts = (
            text.count('"'), text.count("“"), text.count("”"),
            text.count("'"), text.count("‘"), text.count("’"),
            text.count("—"), text.count("–"), text.count("−"),
            text.count("..."),
        )
        extra[row] = np.asarray(counts, dtype=np.float64) * rate
    matrix = np.column_stack((original, extra))
    if not np.isfinite(matrix).all():
        raise ValueError("Fingerprint features contain non-finite values")
    return matrix


def make_variant_pipeline(variant: str):
    model = make_pipeline()
    if variant == "punctuation":
        # The style block still has one StandardScaler and one row Normalizer.
        # Word/character TF-IDF and the SVM are identical to E0c.
        style = dict(model.named_steps["features"].transformer_list)["style"]
        style.named_steps["extract"].func = extract_punctuation_features
    elif variant == "function_words":
        model.named_steps["features"].transformer_list.append((
            "function_words",
            TfidfVectorizer(
                preprocessor=function_word_sequence,
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 3),
                min_df=2,
                max_features=5_000,
                sublinear_tf=True,
                lowercase=False,
                dtype=np.float32,
            ),
        ))
    elif variant != "baseline":
        raise ValueError(f"Unknown variant: {variant}")
    return model


def run(train: Path, e0_dir: Path, e0c_dir: Path, output_dir: Path, variant: str) -> None:
    if output_dir.resolve() in {e0_dir.resolve(), e0c_dir.resolve()}:
        raise ValueError("Choose a new output directory for the ablation")
    frame = load_and_validate(train)
    dev, manifest, e0, folds, _ = load_e0_split(frame, e0_dir, train)
    saved = pd.read_csv(e0c_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    saved_metrics = json.loads((e0c_dir / "metrics.json").read_text(encoding="utf-8"))
    if saved_metrics["train_sha256"] != sha256(train):
        raise ValueError("Saved E0c used a different training file")
    for key in ("id", "prompt_id", "true_model", "fold"):
        if key not in saved or not saved[key].equals(e0[key]):
            raise ValueError(f"Saved E0c {key} differs from the fixed E0 split")
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Expected the locked 2,160-row, 120-prompt development set")

    labels = dev["model"].to_numpy()
    texts = dev["text"].to_numpy()
    predicted = np.full(len(dev), None, dtype=object)
    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        fit_prompts = set(dev.loc[fit_idx, "prompt_id"])
        valid_prompts = set(dev.loc[valid_idx, "prompt_id"])
        if fit_prompts & valid_prompts or set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} has leaked prompts or missing classes")
        model = make_variant_pipeline(variant).fit(texts[fit_idx], labels[fit_idx])
        predicted[valid_idx] = model.predict(texts[valid_idx])
        class_order = model.named_steps["classifier"].classes_
        positions = [int(np.flatnonzero(class_order == label)[0]) for label in LABELS]
        scores[valid_idx] = model.decision_function(texts[valid_idx])[:, positions]
        fold_rows.append({
            "fold": fold,
            "train_rows": int(len(fit_idx)),
            "validation_rows": int(len(valid_idx)),
            "train_prompts": len(fit_prompts),
            "validation_prompts": len(valid_prompts),
            "balanced_accuracy": float(balanced_accuracy_score(labels[valid_idx], predicted[valid_idx])),
            "function_word_vocabulary_size": (
                len(dict(model.named_steps["features"].transformer_list)["function_words"].vocabulary_)
                if variant == "function_words" else None
            ),
        })
        print(f"{variant} fold {fold}/{N_FOLDS}: {fold_rows[-1]['balanced_accuracy']:.6f}", flush=True)

    if any(value is None for value in predicted) or not np.isfinite(scores).all():
        raise RuntimeError("OOF predictions or scores are incomplete")
    score = float(balanced_accuracy_score(labels, predicted))
    saved_score = float(balanced_accuracy_score(labels, saved["pred_model"]))
    if not np.isclose(saved_score, saved_metrics["oof_balanced_accuracy"]):
        raise ValueError("Saved E0c predictions and metrics disagree")
    saved_match = bool(np.array_equal(predicted, saved["pred_model"].to_numpy()))
    confusion = confusion_matrix(labels, predicted, labels=LABELS)
    saved_confusion = confusion_matrix(labels, saved["pred_model"], labels=LABELS)
    support = confusion.sum(axis=1)
    class_metrics = pd.DataFrame({
        "model": LABELS,
        "support": support,
        "correct": confusion.diagonal(),
        "recall": confusion.diagonal() / support,
        "saved_e0c_recall": saved_confusion.diagonal() / support,
    })
    class_metrics["difference_vs_saved_e0c"] = class_metrics["recall"] - class_metrics["saved_e0c_recall"]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["pred_model"] = predicted
    oof["fold"] = folds
    oof["correct"] = labels == predicted
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    decision_scores = dev.loc[:, ["id", "prompt_id"]].copy()
    decision_scores["true_model"] = labels
    decision_scores["fold"] = folds
    for index, label in enumerate(LABELS):
        decision_scores[f"score_{label}"] = scores[:, index]
    decision_scores.to_csv(output_dir / "decision_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(output_dir / "confusion_matrix.csv")
    metrics = {
        "experiment": "e8_fingerprint_ablation",
        "variant": variant,
        "metric": "balanced_accuracy",
        "oof_balanced_accuracy": score,
        "saved_e0c_balanced_accuracy": saved_score,
        "difference_vs_saved_e0c": score - saved_score,
        "predictions_match_saved_e0c": saved_match,
        "n_rows_development": len(dev),
        "n_prompts_development": int(dev["prompt_id"].nunique()),
        "n_prompts_holdout_unscored": int(manifest["split"].eq("holdout").sum()),
        "n_folds": N_FOLDS,
        "fold_metrics": fold_rows,
        "style_feature_names": list(FEATURE_NAMES) + (list(PUNCTUATION_NAMES) if variant == "punctuation" else []),
        "extra_feature_block": "function-word 1-3 gram TF-IDF" if variant == "function_words" else None,
        "train_sha256": sha256(train),
        "e0_manifest_sha256": sha256(e0_dir / "split_manifest.csv"),
        "e0_oof_sha256": sha256(e0_dir / "oof_predictions.csv"),
        "saved_e0c_oof_sha256": sha256(e0c_dir / "oof_predictions.csv"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "holdout_evaluated": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{variant} OOF balanced accuracy: {score:.6f} (saved E0c {saved_score:.6f})", flush=True)
    print(f"Saved E0c predictions reproduced exactly: {saved_match}", flush=True)
    print(f"Artifacts: {output_dir.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0c-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "punctuation", "function_words"), required=True)
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.e0c_dir, args.output_dir, args.variant)


if __name__ == "__main__":
    main()
