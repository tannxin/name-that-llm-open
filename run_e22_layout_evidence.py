#!/usr/bin/env python3
"""E22 preregistered low-confidence, fold-fitted layout evidence.

Protocol: docs/e22_layout_evidence.md. This is one fixed OOF ablation;
validation labels never enter the feature evidence fit or gating rule.
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
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


TRAIN_PATH = Path("data/train.csv")
E0_DIR = Path("outputs/e0_tfidf_svm")
E0C_PATH = Path("outputs/e0c_tfidf_plus_style/decision_scores.csv")
ROBERTA_PATH = Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv")
MODERN_PATHS = {
    42: Path("outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv"),
    17: Path("outputs/e16_modernbert_seed17_eager/oof_predictions.csv"),
    73: Path("outputs/e16_modernbert_seed73_eager/oof_predictions.csv"),
}
E16_PATH = Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv")
OUTPUT_DIR = Path("outputs/e22_layout_evidence")
EXPECTED_TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"

PRIOR_STRENGTH = 48.0
GLOBAL_PRIOR = 1.0
LOG_RATIO_CLIP = 1.5
EVIDENCE_WEIGHT = 0.20
MIN_CHARACTERS = 200
MIN_MARKERS = 2
MAX_BASE_PROBABILITY = 0.30

# Each expression is a single, preregistered binary marker. re.MULTILINE
# applies only to line-anchored markers; all are interpreted on raw text.
MARKERS = (
    ("ascii_double_quote", r'"'),
    ("curly_double_quote", r"[“”]"),
    ("curly_single_quote", r"[‘’]"),
    ("long_dash", r"[—–]"),
    ("ellipsis", r"…|\.{3,}"),
    ("semicolon", r";"),
    ("colon", r":"),
    ("bullet_line", r"^[ \t]*[-*•][ \t]+"),
    ("numbered_line", r"^[ \t]*\d+[.)][ \t]+"),
    ("heading_line", r"^[ \t]*#{1,6}[ \t]+"),
    ("code_fence", r"`{3}|~{3}"),
    ("inline_backtick", r"(?<!`)`(?!`)"),
    ("bold_marker", r"\*\*|__"),
    ("link_or_url", r"\[[^\]\n]+\]\([^)]+\)|https?://"),
    ("table_row", r"^[ \t]*\|[^\n]*\|[^\n]*\|"),
    ("blockquote_line", r"^[ \t]*>[ \t]+"),
    ("math_delimiter", r"\\\(|\\\[|\$[^$\n]+\$"),
)
PATTERNS = tuple((name, re.compile(expr, re.MULTILINE)) for name, expr in MARKERS)


def marker_matrix(texts: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(texts), len(PATTERNS)), dtype=np.uint8)
    for row, value in enumerate(texts):
        for column, (_, pattern) in enumerate(PATTERNS):
            matrix[row, column] = pattern.search(value) is not None
    return matrix


def read_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    if sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA:
        raise ValueError("Official train.csv SHA-256 changed")
    frame = load_and_validate(TRAIN_PATH)
    dev, _, e0, folds, _ = load_e0_split(frame, E0_DIR, TRAIN_PATH)
    paths = {"e0c": E0C_PATH, "roberta": ROBERTA_PATH,
             **{f"modern_{seed}": path for seed, path in MODERN_PATHS.items()}}
    scores = {name: load_scores(path) for name, path in paths.items()}
    reference = scores["e0c"]
    saved_e16 = pd.read_csv(E16_PATH, dtype=str, keep_default_na=False)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Development split is not 2,160 rows / 120 prompts")
    for name, item in {**scores, "saved_e16": saved_e16, "e0": e0}.items():
        if len(item) != len(dev):
            raise ValueError(f"{name} row count differs from development split")
        for key in KEY_COLUMNS:
            if not item[key].equals(reference[key]):
                raise ValueError(f"{name} differs from E0c on {key}")
    if not np.array_equal(folds.astype(str), reference["fold"].to_numpy()):
        raise ValueError("Saved fold assignments changed")
    true = dev["model"].to_numpy()
    if not np.array_equal(true, reference["true_model"].to_numpy()):
        raise ValueError("Development labels changed")
    per_prompt = reference.groupby("prompt_id")["true_model"].agg(list)
    if any(len(group) != len(LABELS) or set(group) != set(LABELS) for group in per_prompt):
        raise ValueError("Each development prompt must contain every class exactly once")
    if any((true == label).sum() != 120 for label in LABELS):
        raise ValueError("Class support differs from 120")

    z = {name: row_standardize(item.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
         for name, item in scores.items()}
    fused = z["e0c"] + z["roberta"] + np.mean(
        np.stack([z[f"modern_{seed}"] for seed in (42, 17, 73)]), axis=0
    )
    base_pred = np.asarray(LABELS)[fused.argmax(axis=1)]
    if not np.array_equal(base_pred, saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("All 2,160 saved E16 OOF predictions were not reproduced")
    if not np.isclose(balanced_accuracy_score(true, base_pred), 0.40925925925925927, atol=1e-12):
        raise ValueError("E16 balanced accuracy failed reproduction")
    hashes = {name: sha256(path) for name, path in paths.items()}
    hashes.update({"train": sha256(TRAIN_PATH), "e16_saved": sha256(E16_PATH)})
    return dev, folds, true, fused, hashes


def softmax(values: np.ndarray) -> np.ndarray:
    centered = values - values.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def run(output_dir: Path) -> dict:
    dev, folds, true, fused, hashes = read_inputs()
    texts = dev["text"].to_numpy()
    markers = marker_matrix(texts)
    lengths = np.fromiter((len(text) for text in texts), dtype=np.int32, count=len(texts))
    marker_counts = markers.sum(axis=1)
    base_logits = fused / 3.0
    max_prob = softmax(base_logits).max(axis=1)
    candidate_logits = base_logits.copy()
    used = np.zeros(len(dev), dtype=bool)
    fit_rows = []

    for fold in range(1, N_FOLDS + 1):
        train = folds != fold
        valid = folds == fold
        if int(train.sum()) != 1728 or int(valid.sum()) != 432:
            raise ValueError(f"Fold {fold} counts changed")
        if set(dev.loc[train, "prompt_id"]) & set(dev.loc[valid, "prompt_id"]):
            raise ValueError(f"Fold {fold} overlaps prompt groups")
        if int(dev.loc[train, "prompt_id"].nunique()) != 96:
            raise ValueError(f"Fold {fold} training prompt count changed")
        x_train = markers[train].astype(np.float64)
        global_rate = (x_train.sum(axis=0) + GLOBAL_PRIOR) / (len(x_train) + 2 * GLOBAL_PRIOR)
        theta = np.empty((len(LABELS), len(PATTERNS)), dtype=np.float64)
        for class_index, label in enumerate(LABELS):
            class_mask = true[train] == label
            if int(class_mask.sum()) != 96:
                raise ValueError(f"Fold {fold} class {label} has wrong training support")
            theta[class_index] = (x_train[class_mask].sum(axis=0) + PRIOR_STRENGTH * global_rate) / (
                int(class_mask.sum()) + PRIOR_STRENGTH
            )
        log_ratio = np.clip(np.log(theta / global_rate), -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
        fold_valid_idx = np.flatnonzero(valid)
        active = (lengths[valid] >= MIN_CHARACTERS) & (marker_counts[valid] >= MIN_MARKERS) & (
            max_prob[valid] < MAX_BASE_PROBABILITY
        )
        valid_idx = fold_valid_idx[active]
        if len(valid_idx):
            evidence = markers[valid_idx].astype(np.float64) @ log_ratio.T
            evidence /= marker_counts[valid_idx, None]
            candidate_logits[valid_idx] += EVIDENCE_WEIGHT * evidence
            used[valid_idx] = True
        fit_rows.append({
            "fold": fold,
            "train_prompts": 96,
            "validation_prompts": 24,
            "train_rows": int(train.sum()),
            "validation_rows": int(valid.sum()),
            "eligible_rows": int(active.sum()),
            "length_ok_rows": int((lengths[valid] >= MIN_CHARACTERS).sum()),
            "marker_ok_rows": int((marker_counts[valid] >= MIN_MARKERS).sum()),
            "low_confidence_rows": int((max_prob[valid] < MAX_BASE_PROBABILITY).sum()),
            "global_marker_rate_min": float(global_rate.min()),
            "global_marker_rate_max": float(global_rate.max()),
        })

    base_pred = np.asarray(LABELS)[base_logits.argmax(axis=1)]
    new_pred = np.asarray(LABELS)[candidate_logits.argmax(axis=1)]
    if np.any((~used) & (base_pred != new_pred)):
        raise ValueError("A prediction changed outside the preregistered eligibility gate")
    folds_out = []
    for fold in range(1, N_FOLDS + 1):
        mask = folds == fold
        old = float(balanced_accuracy_score(true[mask], base_pred[mask]))
        new = float(balanced_accuracy_score(true[mask], new_pred[mask]))
        folds_out.append({"fold": fold, "e16": old, "e22": new, "delta": new - old,
                          "eligible_rows": int(used[mask].sum()),
                          "changed_predictions": int((base_pred[mask] != new_pred[mask]).sum())})
    classes_out = []
    for label in LABELS:
        mask = true == label
        old = float((base_pred[mask] == label).mean())
        new = float((new_pred[mask] == label).mean())
        classes_out.append({"model": label, "support": int(mask.sum()),
                            "e16_recall": old, "e22_recall": new, "delta": new - old})
    fold_table = pd.DataFrame(folds_out)
    class_table = pd.DataFrame(classes_out)
    old_ba = float(balanced_accuracy_score(true, base_pred))
    new_ba = float(balanced_accuracy_score(true, new_pred))
    gate = {
        "pooled_delta_at_least_0_01": bool(new_ba - old_ba >= 0.01 - 1e-12),
        "at_least_four_folds_non_decrease": bool((fold_table["delta"] >= -1e-12).sum() >= 4),
        "all_class_drops_less_than_0_05": bool(class_table["delta"].min() > -0.05 + 1e-12),
    }
    metrics = {
        "experiment": "e22_layout_evidence",
        "status": "completed",
        "input_sha256": hashes,
        "n_rows": len(dev), "n_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": N_FOLDS, "n_markers": len(MARKERS),
        "marker_names": [name for name, _ in MARKERS],
        "marker_hit_counts": {name: int(markers[:, j].sum()) for j, (name, _) in enumerate(MARKERS)},
        "eligible_rows": int(used.sum()),
        "changed_predictions": int((base_pred != new_pred).sum()),
        "wrong_to_right": int(((base_pred != true) & (new_pred == true)).sum()),
        "right_to_wrong": int(((base_pred == true) & (new_pred != true)).sum()),
        "e16_balanced_accuracy": old_ba, "e22_balanced_accuracy": new_ba,
        "delta": new_ba - old_ba,
        "paired_prompt_bootstrap": paired_prompt_bootstrap(
            true, dev["prompt_id"].to_numpy(), base_pred, new_pred
        ),
        "folds_non_decrease": int((fold_table["delta"] >= -1e-12).sum()),
        "worst_class_delta": float(class_table["delta"].min()),
        "go_criteria": gate, "go": all(gate.values()),
        "saved_e16_predictions_exactly_reproduced": True,
        "parameters": {
            "prior_strength": PRIOR_STRENGTH, "global_prior": GLOBAL_PRIOR,
            "log_ratio_clip": LOG_RATIO_CLIP, "evidence_weight": EVIDENCE_WEIGHT,
            "min_characters": MIN_CHARACTERS, "min_markers": MIN_MARKERS,
            "max_base_probability_strict": MAX_BASE_PROBABILITY,
        },
        "holdout_evaluated": False, "weights_tuned": False,
        "python_version": sys.version.split()[0], "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(output_dir / "fold_metrics.csv", index=False)
    class_table.to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "fit_diagnostics.csv", index=False)
    oof = dev.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = true
    oof["fold"] = folds
    oof["e16_pred"] = base_pred
    oof["e22_pred"] = new_pred
    oof["eligible"] = used
    oof["marker_count"] = marker_counts
    oof["char_length"] = lengths
    oof["e16_max_probability"] = max_prob
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
