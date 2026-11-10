#!/usr/bin/env python3
"""E7 diagnostic: classify model family, then exact version within the family.

Uses E0c's unchanged feature blocks and E0's prompt-grouped folds. The locked
30-prompt holdout is excluded from fitting, feature extraction, and scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import load_e0_split


def family(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unknown official label: {label}")
    if label.startswith("qwen"):
        return "qwen"
    return label.split("-", 1)[0]


FAMILIES = tuple(sorted({family(label) for label in LABELS}))


def new_head() -> LinearSVC:
    return LinearSVC(C=1.0, dual=True, max_iter=10_000, random_state=42)


def fit_cascade(texts: np.ndarray, labels: np.ndarray):
    features = make_pipeline().named_steps["features"]
    matrix = features.fit_transform(texts)
    families = np.asarray([family(label) for label in labels])
    family_head = new_head().fit(matrix, families)
    version_heads = {}
    for name in FAMILIES:
        family_labels = tuple(label for label in LABELS if family(label) == name)
        if len(family_labels) == 1:
            version_heads[name] = family_labels[0]
        else:
            version_heads[name] = new_head().fit(matrix[families == name], labels[families == name])
    return features, family_head, version_heads


def predict_cascade(bundle, texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features, family_head, version_heads = bundle
    matrix = features.transform(texts)
    predicted_family = family_head.predict(matrix)
    predicted = np.empty(len(texts), dtype=object)
    for name in FAMILIES:
        indices = np.flatnonzero(predicted_family == name)
        if not len(indices):
            continue
        head = version_heads[name]
        predicted[indices] = head if isinstance(head, str) else head.predict(matrix[indices])
    return predicted_family, predicted


def run(train: Path, e0_dir: Path, e0c_dir: Path, output_dir: Path) -> None:
    frame = load_and_validate(train)
    development, manifest, e0, folds, e0_metrics = load_e0_split(frame, e0_dir, train)
    e0c = pd.read_csv(e0c_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for column in ("id", "prompt_id", "true_model", "fold"):
        if not e0c[column].equals(e0[column]):
            raise ValueError(f"E0c {column} differs from the saved E0 fold assignment")
    if output_dir.resolve() in {e0_dir.resolve(), e0c_dir.resolve()}:
        raise ValueError("Output directory must differ from the baseline directories")

    texts = development["text"].to_numpy()
    labels = development["model"].to_numpy()
    families = np.asarray([family(label) for label in labels])
    predicted = np.empty(len(development), dtype=object)
    predicted_family = np.empty(len(development), dtype=object)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks one or more exact model labels")
        if set(development.loc[fit_idx, "prompt_id"]) & set(development.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping prompts")
        bundle = fit_cascade(texts[fit_idx], labels[fit_idx])
        predicted_family[valid_idx], predicted[valid_idx] = predict_cascade(bundle, texts[valid_idx])
        score = balanced_accuracy_score(labels[valid_idx], predicted[valid_idx])
        e0c_score = balanced_accuracy_score(labels[valid_idx], e0c.loc[valid_idx, "pred_model"])
        fold_rows.append({
            "fold": fold,
            "train_rows": len(fit_idx),
            "validation_rows": len(valid_idx),
            "balanced_accuracy": float(score),
            "e0c_balanced_accuracy": float(e0c_score),
            "family_accuracy": float(accuracy_score(families[valid_idx], predicted_family[valid_idx])),
        })
        print(f"fold={fold} hierarchical={score:.6f} E0c={e0c_score:.6f}", flush=True)
    if any(value is None for value in predicted):
        raise RuntimeError("Incomplete OOF predictions")

    score = float(balanced_accuracy_score(labels, predicted))
    e0c_score = float(balanced_accuracy_score(labels, e0c["pred_model"]))
    confusion = confusion_matrix(labels, predicted, labels=LABELS)
    class_metrics = pd.DataFrame({
        "model": LABELS,
        "support": confusion.sum(axis=1),
        "correct": confusion.diagonal(),
    })
    class_metrics["recall"] = class_metrics["correct"] / class_metrics["support"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    oof = development[["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["pred_model"] = predicted
    oof["true_family"] = families
    oof["pred_family"] = predicted_family
    oof["fold"] = folds
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(output_dir / "confusion_matrix.csv")
    metrics = {
        "experiment": "e7_hierarchical_diagnostic",
        "oof_balanced_accuracy": score,
        "e0c_oof_balanced_accuracy": e0c_score,
        "difference_vs_e0c": score - e0c_score,
        "oof_family_accuracy": float(accuracy_score(families, predicted_family)),
        "fold_metrics": fold_rows,
        "train_sha256": sha256(train),
        "n_rows_development": len(development),
        "n_prompts_holdout_untouched": int(manifest["split"].eq("holdout").sum()),
        "n_folds": N_FOLDS,
        "feature_blocks": "same word, character, and style blocks as E0c",
        "classifier": "hard family LinearSVC then within-family LinearSVC (C=1); singleton families use their sole label",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(fit_cascade(texts, labels), output_dir / "dev_model.joblib", compress=3)
    print(f"OOF hierarchical={score:.6f} E0c={e0c_score:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0c-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e7_hierarchical"))
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.e0c_dir, args.output_dir)


if __name__ == "__main__":
    main()
