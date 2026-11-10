#!/usr/bin/env python3
"""E3 exploratory augmentation from a provenance-filtered public dataset.

Only eight labels with unambiguous served model IDs are considered. External
examples join each fold's training rows; official validation prompts and the
locked 30-prompt holdout never enter the fitted classifier. A high-threshold
near-duplicate screen is an audit, not a claim that prompts cannot overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_style import load_e0_split


EXPECTED_ZIP_SHA256 = "f7fd25a04fb54a0a365906089047283286c6d9a7b7ebdc954dc826b4083f622a"
EXACT_MODELS = {
    "gpt-5.5": ("openai/gpt-5.5", "OpenAI"),
    "gpt-5.6-luna": ("openai/gpt-5.6-luna", "OpenAI"),
    "gpt-5.6-sol": ("openai/gpt-5.6-sol", "OpenAI"),
    "gpt-5.6-terra": ("openai/gpt-5.6-terra", "OpenAI"),
    "gpt-6-astra": ("openai/gpt-6-astra", "OpenAI"),
    "glm-5.3": ("z-ai/glm-5.3", "Z.AI"),
    "glm-5.3-flash": ("z-ai/glm-5.3-flash", "Z.AI"),
    "kimi-k3": ("moonshotai/kimi-k3", "Moonshot AI"),
}
MAX_EXTERNAL_PER_CLASS = 250
NEAR_DUP_THRESHOLD = 0.90
EXTERNAL_MASS_RATIOS = (0.25, 1.0)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def load_external(path: Path, development: pd.DataFrame) -> pd.DataFrame:
    if sha256(path) != EXPECTED_ZIP_SHA256:
        raise ValueError("Will Rice ZIP SHA-256 differs from audited version 5")
    with zipfile.ZipFile(path) as archive:
        external = pd.read_csv(
            io.BytesIO(archive.read("passages.csv")),
            dtype=str, keep_default_na=False,
        )
        prompts = pd.read_csv(
            io.BytesIO(archive.read("prompts.csv")),
            dtype=str, keep_default_na=False,
        )
    required = {
        "id", "label", "requested_model", "served_model", "served_provider",
        "finish_reason", "prompt_id", "prompt", "text",
    }
    if not required.issubset(external.columns):
        raise ValueError("External passage columns differ from audit")
    if len(external) != 10_772 or external["id"].duplicated().any():
        raise ValueError("External row count or IDs differ from audit")
    if prompts["prompt_id"].duplicated().any() or len(prompts) != 1000:
        raise ValueError("External prompt table differs from audit")
    if (external["requested_model"] != external["served_model"]).any():
        raise ValueError("Requested and served model IDs differ")
    if not external["finish_reason"].eq("stop").all():
        raise ValueError("Some external completions did not stop normally")
    if external[["id", "prompt_id", "prompt", "text"]].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("External IDs, prompts, or text contain blanks")
    selected = external.loc[external["label"].isin(EXACT_MODELS)].copy()
    for label, (served_model, provider) in EXACT_MODELS.items():
        rows = selected["label"].eq(label)
        if not selected.loc[rows, "served_model"].eq(served_model).all():
            raise ValueError(f"Unexpected served model for {label}")
        if not selected.loc[rows, "served_provider"].eq(provider).all():
            raise ValueError(f"Unexpected served provider for {label}")
        if rows.sum() < MAX_EXTERNAL_PER_CLASS:
            raise ValueError(f"Too few external rows for {label}")

    # Exact normalized duplicates are disallowed before the similarity screen.
    selected["normalized_text"] = selected["text"].map(normalize_text)
    development_texts = set(development["text"].map(normalize_text))
    if selected["normalized_text"].duplicated().any():
        raise ValueError("External exact duplicate text found")
    if selected["normalized_text"].isin(development_texts).any():
        raise ValueError("External text exactly duplicates development data")
    return selected.reset_index(drop=True)


def near_duplicate_screen(development: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    # A fixed, sparse char-4 representation gives a reproducible high-similarity
    # screen without comparing the 10k rows using quadratic edit distance.
    texts = development["text"].tolist() + external["text"].tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(4, 4), min_df=2,
        max_features=100_000, lowercase=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    n_development = len(development)
    dev_features = matrix[:n_development]
    maxima = np.zeros(len(external), dtype=np.float32)
    for label in EXACT_MODELS:
        ext_idx = np.flatnonzero(external["label"].eq(label).to_numpy())
        # Compare against every development response without reading its label.
        # The final holdout rows are outside this function entirely.
        similarities = (matrix[n_development + ext_idx] @ dev_features.T).toarray()
        maxima[ext_idx] = similarities.max(axis=1)
        print(
            f"near-duplicate screen {label}: max={maxima[ext_idx].max():.4f}, "
            f"flagged={(maxima[ext_idx] >= NEAR_DUP_THRESHOLD).sum()}",
            flush=True,
        )
    report = external[["id", "label", "served_model", "served_provider", "prompt_id"]].copy()
    report["max_char4_cosine_to_development"] = maxima
    report["excluded_near_duplicate"] = maxima >= NEAR_DUP_THRESHOLD
    return report


def sample_external(external: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    usable = external.loc[~report["excluded_near_duplicate"].to_numpy()].copy()
    usable["sample_key"] = usable["id"].map(
        lambda value: hashlib.sha256(("42:" + value).encode("utf-8")).hexdigest()
    )
    sampled = []
    for label in EXACT_MODELS:
        rows = usable.loc[usable["label"].eq(label)].sort_values("sample_key")
        if len(rows) < MAX_EXTERNAL_PER_CLASS:
            raise ValueError(f"Only {len(rows)} usable external rows for {label}")
        sampled.append(rows.head(MAX_EXTERNAL_PER_CLASS))
    return pd.concat(sampled, ignore_index=True)


def sample_weights(official_labels: np.ndarray, external_labels: np.ndarray, ratio: float) -> np.ndarray:
    weights_off = np.ones(len(official_labels), dtype=np.float32)
    weights_ext = np.zeros(len(external_labels), dtype=np.float32)
    for label in EXACT_MODELS:
        off_mask = official_labels == label
        ext_mask = external_labels == label
        n_off, n_ext = int(off_mask.sum()), int(ext_mask.sum())
        if not n_off or not n_ext:
            raise ValueError(f"Missing official or external examples for {label}")
        weights_off[off_mask] = 1.0 / (1.0 + ratio)
        weights_ext[ext_mask] = ratio * n_off / (n_ext * (1.0 + ratio))
    return np.concatenate((weights_off, weights_ext))


def run(train: Path, e0_dir: Path, e0c_dir: Path, external_zip: Path, output_dir: Path) -> None:
    official = load_and_validate(train)
    dev, manifest, e0, folds, _ = load_e0_split(official, e0_dir, train)
    e0c = pd.read_csv(e0c_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    for column in ("id", "prompt_id", "true_model", "fold"):
        if not e0c[column].equals(e0[column]):
            raise ValueError(f"E0c {column} differs from E0")
    ext = load_external(external_zip, dev)
    audit = near_duplicate_screen(dev, ext)
    sampled = sample_external(ext, audit)
    print(f"selected {len(sampled)} external rows across {len(EXACT_MODELS)} labels", flush=True)

    texts = dev["text"].to_numpy()
    labels = dev["model"].to_numpy()
    ext_texts = sampled["text"].to_numpy()
    ext_labels = sampled["label"].to_numpy()
    predictions = {ratio: np.empty(len(dev), dtype=object) for ratio in EXTERNAL_MASS_RATIOS}
    fold_scores = {ratio: [] for ratio in EXTERNAL_MASS_RATIOS}
    for fold in range(1, N_FOLDS + 1):
        fit_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        if set(dev.loc[fit_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Fold {fold} has overlapping official prompts")
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks an official class")
        training_texts = np.concatenate((texts[fit_idx], ext_texts))
        training_labels = np.concatenate((labels[fit_idx], ext_labels))
        features = make_pipeline().named_steps["features"]
        x_fit = features.fit_transform(training_texts)
        x_valid = features.transform(texts[valid_idx])
        for ratio in EXTERNAL_MASS_RATIOS:
            head = LinearSVC(
                C=1.0, dual=True, max_iter=10_000, random_state=42,
            )
            weights = sample_weights(labels[fit_idx], ext_labels, ratio)
            head.fit(x_fit, training_labels, sample_weight=weights)
            pred = head.predict(x_valid)
            predictions[ratio][valid_idx] = pred
            fold_scores[ratio].append(float(balanced_accuracy_score(labels[valid_idx], pred)))
        print(f"E3 completed fold {fold}/{N_FOLDS}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "external_near_duplicate_audit.csv", index=False)
    sampled[["id", "label", "served_model", "served_provider", "prompt_id"]].to_csv(
        output_dir / "selected_external_manifest.csv", index=False,
    )
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    scores = {}
    for ratio, pred in predictions.items():
        if any(value is None for value in pred):
            raise RuntimeError(f"Incomplete OOF predictions for external ratio {ratio}")
        slug = str(ratio).replace(".", "p")
        oof = dev[["id", "prompt_id"]].copy()
        oof["true_model"] = labels
        oof["pred_model"] = pred
        oof["fold"] = folds
        oof.to_csv(output_dir / f"oof_ratio_{slug}.csv", index=False)
        matrix = confusion_matrix(labels, pred, labels=LABELS)
        class_metrics = pd.DataFrame({
            "model": LABELS,
            "support": matrix.sum(axis=1),
            "correct": matrix.diagonal(),
            "recall": matrix.diagonal() / matrix.sum(axis=1),
        })
        class_metrics.to_csv(output_dir / f"class_metrics_ratio_{slug}.csv", index=False)
        scores[str(ratio)] = {
            "oof_balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "fold_balanced_accuracy": fold_scores[ratio],
        }
    metrics = {
        "experiment": "e3_willrice_exploratory",
        "official_train_sha256": sha256(train),
        "external_zip_sha256": sha256(external_zip),
        "external_dataset": "Will Rice OpenRouter Harvest, Kaggle version 5, CC BY 4.0",
        "included_labels": list(EXACT_MODELS),
        "excluded_labels": "Gemini, Qwen, Claude: snapshot ambiguity or no rows",
        "near_duplicate_method": "all-label char-4 TF-IDF cosine against development rows only",
        "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        "near_duplicate_excluded_rows": int(audit["excluded_near_duplicate"].sum()),
        "sampled_external_rows_per_label": MAX_EXTERNAL_PER_CLASS,
        "external_mass_ratio_to_official_per_supported_class": EXTERNAL_MASS_RATIOS,
        "official_only_e0c_oof_balanced_accuracy": float(
            balanced_accuracy_score(labels, e0c["pred_model"])
        ),
        "external_prompt_overlap_with_official_unknown": True,
        "same_folds_as_e0": True,
        "holdout_labels_not_used": True,
        "scores": scores,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e0c-dir", type=Path, default=Path("outputs/e0c_tfidf_plus_style"))
    parser.add_argument("--external-zip", type=Path, default=Path("data/external/willrice/source_v5.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e3_willrice"))
    args = parser.parse_args()
    run(args.train, args.e0_dir, args.e0c_dir, args.external_zip, args.output_dir)


if __name__ == "__main__":
    main()
