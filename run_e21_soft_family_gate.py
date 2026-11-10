#!/usr/bin/env python3
"""E21 preregistered probability-conserving soft family experts.

Protocol: docs/e21_soft_family_gate.md. One recipe, no validation tuning.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score

from analyze_modernbert_seeds import family
from run_combined import make_pipeline
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
OUTPUT_DIR = Path("outputs/e21_soft_family_gate")
EXPECTED_TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
GATE_MAX = 0.25
FAMILIES = {
    "gpt": LABELS[0:5],
    "gemini": LABELS[5:9],
    "claude": LABELS[9:14],
    "qwen": LABELS[14:15],
    "glm": LABELS[15:17],
    "kimi": LABELS[17:18],
}


def softmax(values: np.ndarray) -> np.ndarray:
    centered = values - values.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def expert_scores(model: sklearn.pipeline.Pipeline, texts: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    classes = np.asarray(classifier.classes_)
    raw = np.asarray(model.decision_function(texts), dtype=np.float64)
    if len(classes) == 2:
        if raw.ndim != 1:
            raise ValueError("Binary LinearSVC did not produce one-dimensional margins")
        # sklearn's binary margin is positive for classes_[1].
        raw = np.column_stack((-raw, raw))
    elif raw.shape != (len(texts), len(classes)):
        raise ValueError("Multiclass LinearSVC has an unexpected score shape")
    if not np.array_equal(classes[raw.argmax(axis=1)], model.predict(texts)):
        raise ValueError("Expanded expert margins disagree with sklearn predictions")
    if set(classes) != set(labels):
        raise ValueError("Expert class set differs from the preregistered family")
    positions = [int(np.flatnonzero(classes == label)[0]) for label in labels]
    return raw[:, positions]


def expert_probabilities(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    standardized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)
    return softmax(standardized)


def read_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    if sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA:
        raise ValueError("Official training data changed")
    frame = load_and_validate(TRAIN_PATH)
    dev, _, e0, folds, _ = load_e0_split(frame, E0_DIR, TRAIN_PATH)
    sources = {"e0c": E0C_PATH, "roberta": ROBERTA_PATH,
               **{f"modern_{seed}": path for seed, path in MODERN_PATHS.items()}}
    loaded = {name: load_scores(path) for name, path in sources.items()}
    reference = loaded["e0c"]
    saved_e16 = pd.read_csv(E16_PATH, dtype=str, keep_default_na=False)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("The original development split changed")
    for name, item in {**loaded, "saved_e16": saved_e16, "e0": e0}.items():
        if len(item) != len(dev):
            raise ValueError(f"{name} length differs from development split")
        for key in KEY_COLUMNS:
            if not item[key].equals(reference[key]):
                raise ValueError(f"{name} differs from E0c on {key}")
    if not np.array_equal(folds.astype(str), reference["fold"].to_numpy()):
        raise ValueError("Original fold membership changed")
    labels = dev["model"].to_numpy()
    if not np.array_equal(labels, reference["true_model"].to_numpy()):
        raise ValueError("Development labels changed")
    prompt_labels = reference.groupby("prompt_id")["true_model"].agg(list)
    if any(len(group) != len(LABELS) or set(group) != set(LABELS) for group in prompt_labels):
        raise ValueError("Each prompt must contain one example of every official label")
    if any((labels == label).sum() != 120 for label in LABELS):
        raise ValueError("Class support differs from 120")
    for name, family_labels in FAMILIES.items():
        if any(family(label) != name for label in family_labels):
            raise ValueError("Family membership changed")

    z = {name: row_standardize(item.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64))
         for name, item in loaded.items()}
    fused = z["e0c"] + z["roberta"] + np.mean(
        np.stack([z[f"modern_{seed}"] for seed in (42, 17, 73)]), axis=0
    )
    predicted = np.asarray(LABELS)[fused.argmax(axis=1)]
    if not np.array_equal(predicted, saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("E16 saved OOF predictions failed exact row-level reproduction")
    score = balanced_accuracy_score(labels, predicted)
    if not np.isclose(score, 0.40925925925925927, atol=1e-12):
        raise ValueError("E16 balanced accuracy failed reproduction")
    hashes = {name: sha256(path) for name, path in sources.items()}
    hashes.update({"train": sha256(TRAIN_PATH), "e16_saved": sha256(E16_PATH)})
    return dev, folds, labels, fused, hashes


def fit_all_experts(dev: pd.DataFrame, folds: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    texts = dev["text"].to_numpy()
    expert_q = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float64)
    fitting = []
    for fold in range(1, N_FOLDS + 1):
        train = folds != fold
        valid = folds == fold
        if set(dev.loc[train, "prompt_id"]) & set(dev.loc[valid, "prompt_id"]):
            raise ValueError("Fold has overlapping prompt groups")
        for name, family_labels in FAMILIES.items():
            if len(family_labels) == 1:
                continue
            family_train = train & np.isin(labels, family_labels)
            if int(family_train.sum()) != 96 * len(family_labels):
                raise ValueError(f"{name} training count differs from preregistered count")
            model = make_pipeline()
            model.fit(texts[family_train], labels[family_train])
            score = expert_scores(model, texts[valid], family_labels)
            cols = [LABELS.index(label) for label in family_labels]
            expert_q[np.ix_(np.flatnonzero(valid), cols)] = expert_probabilities(score)
            fitting.append({"fold": fold, "family": name,
                            "train_rows": int(family_train.sum()),
                            "train_prompts": int(dev.loc[family_train, "prompt_id"].nunique()),
                            "validation_rows": int(valid.sum()),
                            "validation_prompts": int(dev.loc[valid, "prompt_id"].nunique())})
            print(f"fold={fold} family={name} fit_rows={family_train.sum()}", flush=True)
    for name, family_labels in FAMILIES.items():
        if len(family_labels) > 1:
            cols = [LABELS.index(label) for label in family_labels]
            if not np.isfinite(expert_q[:, cols]).all():
                raise ValueError(f"{name} expert OOF incomplete")
            if not np.allclose(expert_q[:, cols].sum(axis=1), 1, atol=1e-12):
                raise ValueError(f"{name} expert probabilities do not sum to one")
    return expert_q, fitting


def apply_gate(fused: np.ndarray, expert_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    base = softmax(fused / 3.0)
    adjusted = base.copy()
    alphas = np.zeros((len(base), len(FAMILIES)), dtype=np.float64)
    errors = {}
    for fi, (name, family_labels) in enumerate(FAMILIES.items()):
        cols = [LABELS.index(label) for label in family_labels]
        mass = base[:, cols].sum(axis=1)
        if len(cols) > 1:
            conditional = base[:, cols] / mass[:, None]
            entropy = -np.sum(conditional * np.log(conditional), axis=1) / np.log(len(cols))
            alpha = GATE_MAX * np.clip(entropy, 0.0, 1.0)
            alphas[:, fi] = alpha
            adjusted[:, cols] = mass[:, None] * (
                (1.0 - alpha[:, None]) * conditional + alpha[:, None] * expert_q[:, cols]
            )
        errors[name] = float(np.max(np.abs(base[:, cols].sum(axis=1)
                                          - adjusted[:, cols].sum(axis=1))))
        if errors[name] > 1e-12:
            raise ValueError(f"{name} family mass not conserved: {errors[name]}")
    errors["all_classes"] = float(np.max(np.abs(adjusted.sum(axis=1) - 1.0)))
    if errors["all_classes"] > 1e-12 or not np.isfinite(adjusted).all():
        raise ValueError("Gated probabilities are invalid")
    return base, adjusted, errors


def report(dev: pd.DataFrame, folds: np.ndarray, labels: np.ndarray, fused: np.ndarray,
           expert_q: np.ndarray, fitting: list[dict], hashes: dict[str, str]) -> None:
    label_array = np.asarray(LABELS)
    base, adjusted, mass_errors = apply_gate(fused, expert_q)
    original = label_array[base.argmax(axis=1)]
    gated = label_array[adjusted.argmax(axis=1)]
    true_families = np.asarray([family(label) for label in labels])
    base_families = np.asarray([family(label) for label in original])
    gated_families = np.asarray([family(label) for label in gated])
    prompts = dev["prompt_id"].to_numpy()

    family_rows = []
    oracle_expert = np.empty(len(labels), dtype=object)
    for name, family_labels in FAMILIES.items():
        cols = [LABELS.index(label) for label in family_labels]
        mask = true_families == name
        base_within = label_array[np.asarray(cols)[base[mask][:, cols].argmax(axis=1)]]
        gated_within = label_array[np.asarray(cols)[adjusted[mask][:, cols].argmax(axis=1)]]
        if len(cols) > 1:
            expert_within = label_array[np.asarray(cols)[expert_q[mask][:, cols].argmax(axis=1)]]
        else:
            expert_within = np.repeat(family_labels[0], int(mask.sum()))
        oracle_expert[mask] = expert_within
        family_rows.append({
            "family": name, "classes": len(cols), "rows": int(mask.sum()),
            "base_family_recall": float((base_families[mask] == name).mean()),
            "gated_family_recall": float((gated_families[mask] == name).mean()),
            "base_within_true_family_accuracy": float((base_within == labels[mask]).mean()),
            "gated_within_true_family_accuracy": float((gated_within == labels[mask]).mean()),
            "expert_within_true_family_accuracy": float((expert_within == labels[mask]).mean()),
            "base_exact_recall": float((original[mask] == labels[mask]).mean()),
            "gated_exact_recall": float((gated[mask] == labels[mask]).mean()),
        })
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        mask = folds == fold
        base_ba = float(balanced_accuracy_score(labels[mask], original[mask]))
        gated_ba = float(balanced_accuracy_score(labels[mask], gated[mask]))
        fold_rows.append({"fold": fold, "base_ba": base_ba,
                          "gated_ba": gated_ba, "delta": gated_ba - base_ba})
    class_rows = []
    for label in LABELS:
        mask = labels == label
        base_recall = float((original[mask] == label).mean())
        gated_recall = float((gated[mask] == label).mean())
        class_rows.append({"model": label, "support": int(mask.sum()),
                           "base_recall": base_recall, "gated_recall": gated_recall,
                           "delta": gated_recall - base_recall})
    prompt_rows = []
    for prompt_id, group in dev.groupby("prompt_id", sort=True):
        idx = group.index.to_numpy()
        base_count = int((original[idx] == labels[idx]).sum())
        gated_count = int((gated[idx] == labels[idx]).sum())
        prompt_rows.append({"prompt_id": prompt_id, "fold": int(folds[idx[0]]),
                            "base_correct": base_count, "gated_correct": gated_count,
                            "delta": gated_count - base_count})
    fold_table = pd.DataFrame(fold_rows)
    class_table = pd.DataFrame(class_rows)
    family_table = pd.DataFrame(family_rows)
    prompt_table = pd.DataFrame(prompt_rows)
    improved = (original != labels) & (gated == labels)
    harmed = (original == labels) & (gated != labels)
    base_family_hit = base_families == true_families
    gated_family_hit = gated_families == true_families
    metrics = {
        "experiment": "e21_soft_family_gate",
        "preregistered_protocol": "docs/e21_soft_family_gate.md",
        "train_sha256": hashes["train"], "input_sha256": hashes,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "e16_oof_exactly_reproduced": True,
        "base_ba": float(balanced_accuracy_score(labels, original)),
        "gated_ba": float(balanced_accuracy_score(labels, gated)),
        "delta_ba": float(balanced_accuracy_score(labels, gated) - balanced_accuracy_score(labels, original)),
        "paired_prompt_bootstrap": paired_prompt_bootstrap(labels, prompts, original, gated),
        "base_family_accuracy": float(base_family_hit.mean()),
        "gated_family_accuracy": float(gated_family_hit.mean()),
        "base_exact_given_correct_family": float((original == labels).sum() / base_family_hit.sum()),
        "gated_exact_given_correct_family": float((gated == labels).sum() / gated_family_hit.sum()),
        "wrong_to_right": int(improved.sum()), "right_to_wrong": int(harmed.sum()),
        "wrong_to_right_with_base_family_correct": int((improved & base_family_hit).sum()),
        "right_to_wrong_with_base_family_correct": int((harmed & base_family_hit).sum()),
        "family_correct_to_wrong": int((base_family_hit & ~gated_family_hit).sum()),
        "family_wrong_to_correct": int((~base_family_hit & gated_family_hit).sum()),
        "max_probability_mass_error": mass_errors,
        "fold_non_decrease": int((fold_table["delta"] >= -1e-12).sum()),
        "worst_class_recall_delta": float(class_table["delta"].min()),
        "prompts_improved": int((prompt_table["delta"] > 0).sum()),
        "prompts_equal": int((prompt_table["delta"] == 0).sum()),
        "prompts_harmed": int((prompt_table["delta"] < 0).sum()),
        "train_fit_metadata": fitting,
        "fusion_weights_tuned": False, "holdout_evaluated": False,
    }
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise ValueError(f"Output directory is not empty: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fold_table.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    class_table.to_csv(OUTPUT_DIR / "class_metrics.csv", index=False)
    family_table.to_csv(OUTPUT_DIR / "family_metrics.csv", index=False)
    prompt_table.to_csv(OUTPUT_DIR / "prompt_metrics.csv", index=False)
    oof = dev[["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["fold"] = folds
    oof["e16_pred_model"] = original
    oof["gated_pred_model"] = gated
    oof["oracle_family_expert_pred_model"] = oracle_expert
    oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    print(json.dumps({key: metrics[key] for key in (
        "base_ba", "gated_ba", "delta_ba", "paired_prompt_bootstrap",
        "wrong_to_right", "right_to_wrong", "max_probability_mass_error",
        "fold_non_decrease", "worst_class_recall_delta")}, indent=2), flush=True)


def main() -> None:
    dev, folds, labels, fused, hashes = read_inputs()
    expert_q, fitting = fit_all_experts(dev, folds, labels)
    report(dev, folds, labels, fused, expert_q, fitting, hashes)


if __name__ == "__main__":
    main()
