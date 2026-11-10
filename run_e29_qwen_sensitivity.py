#!/usr/bin/env python3
"""E29: fixed E24 versus E24 plus audited Qwen 0902 rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score

from analyze_modernbert_seeds import score_columns, validate_keys, validate_run
from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_e24_prvsiyan_external_ablation import (
    AUDIT_KEYS, EXPECTED_SOURCE_SHA256, EXPECTED_TRAIN_SHA256,
    ordered_scores, select_external,
)
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_score_normalization import paired_prompt_bootstrap, row_standardize
from run_style import load_e0_split


QWEN_MANIFEST = Path("data/external/prvsiyan/e29_qwen_accepted_manifest.csv")
QWEN_AUDIT = Path("outputs/e29_qwen_audit/report.json")
E24_AUDIT = Path("data/external/prvsiyan/e24_accepted_manifest.csv")
E24_REPORT = Path("outputs/e24_prvsiyan_audit/report.json")
SOURCE = Path("data/external/prvsiyan/passages.csv")
TRAIN = Path("data/train.csv")
E0_DIR = Path("outputs/e0_tfidf_svm")
E0C_SCORES = Path("outputs/e0c_tfidf_plus_style/decision_scores.csv")
E24_FIT = Path("outputs/e24_prvsiyan_fit")
E24_ANALYSIS = Path("outputs/e24_prvsiyan_analysis")
ROBERTA_SCORES = Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv")
SEED_DIRS = {
    42: Path("outputs/e15_modernbert_base_12ep_eager"),
    17: Path("outputs/e16_modernbert_seed17_eager"),
    73: Path("outputs/e16_modernbert_seed73_eager"),
}
SNAPSHOT = "qwen/qwen3.8-max-0902"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def select_qwen(chosen_prompts: set[str]) -> tuple[pd.DataFrame, dict]:
    report = json.loads(QWEN_AUDIT.read_text(encoding="utf-8"))
    require(report.get("audit_passed") is True and report.get("source_version") == 9
            and report.get("raw_new_qwen_rows") == 300
            and report.get("accepted_rows") == 300
            and report.get("served_model_id") == SNAPSHOT
            and report.get("served_provider") == "Alibaba"
            and report.get("envelopes", {}).get("new_calls_matched") == 300
            and report.get("envelopes", {}).get("mismatched_fields") == 0
            and report.get("envelopes", {}).get("missing_calls") == 0
            and report.get("answer_duplicate_screen", {}).get("count_ge_0_9") == 0
            and report.get("competition_train_derived_source_columns_used") is False,
            "E29 Qwen source audit did not pass registered gates")
    require(report["source_file_sha256"]["passages.csv"] == EXPECTED_SOURCE_SHA256
            and report["accepted_manifest_sha256"] == sha256(QWEN_MANIFEST),
            "E29 source/manifest hash differs from audit")
    manifest = pd.read_csv(QWEN_MANIFEST, dtype=str, keep_default_na=False)
    require(len(manifest) == 300 and manifest["id"].is_unique,
            "Expected all 300 audited Qwen rows")
    require(manifest["source_file_sha256"].eq(EXPECTED_SOURCE_SHA256).all(),
            "Qwen manifest has wrong source hash")
    source = pd.read_csv(
        SOURCE, dtype=str, keep_default_na=False,
        usecols=["id", "prompt_id", "model", "text", "split", "call_id",
                 "provider_route", "served_model_id", "served_provider"],
    )
    selected_raw = source.set_index("id", drop=False).loc[manifest["id"].to_numpy()].reset_index(drop=True)
    for key in AUDIT_KEYS:
        require(np.array_equal(selected_raw[key].to_numpy(), manifest[key].to_numpy()),
                f"Qwen audit key differs from raw source: {key}")
    require(selected_raw["split"].eq("new").all()
            and selected_raw["model"].eq("qwen3.8-max").all()
            and selected_raw["provider_route"].eq("openrouter").all()
            and selected_raw["served_model_id"].eq(SNAPSHOT).all()
            and selected_raw["served_provider"].eq("Alibaba").all(),
            "Qwen route/source identity changed")
    selected = selected_raw.loc[selected_raw["prompt_id"].isin(chosen_prompts)].copy()
    require(len(selected) == 250 and selected["prompt_id"].is_unique,
            "Expected 250 Qwen rows under E24's fixed prompt selection")
    selected = selected.sort_values(["prompt_id", "model", "id"], kind="stable").reset_index(drop=True)
    selected["sample_weight"] = 24.0 / len(selected)
    require(np.isclose(selected["sample_weight"].sum(), 24.0), "Qwen total weight differs from E24")
    return selected, {
        "qwen_audit_report_sha256": sha256(QWEN_AUDIT),
        "qwen_manifest_sha256": sha256(QWEN_MANIFEST),
        "qwen_selected_rows": len(selected),
        "qwen_total_weight_per_fold": 24.0,
        "served_model_id": SNAPSHOT,
        "served_provider": "Alibaba",
    }


def fit(output: Path) -> None:
    require(not output.exists() or not any(output.iterdir()), "E29 fit output is not empty")
    require(sys.version_info[:3] == (3, 10, 12) and sklearn.__version__ == "1.7.2",
            "Use original Python 3.10.12 / scikit-learn 1.7.2 environment")
    require(sha256(TRAIN) == EXPECTED_TRAIN_SHA256, "Official train hash changed")
    e24_external, prompt_manifest, e24_selection = select_external(SOURCE, E24_AUDIT, E24_REPORT)
    require(len(e24_external) == 3620 and sha256(E24_FIT / "selected_external.csv")
            == "d7f8b7c9b0480ec896777e667589bf7d2fe12e503aa8f867c0708d0656ad6191",
            "E24 selected sample changed")
    chosen = set(prompt_manifest.loc[prompt_manifest["selected"], "prompt_id"])
    qwen, qwen_meta = select_qwen(chosen)
    require(set(qwen["id"]).isdisjoint(set(e24_external["id"])), "Qwen and E24 row IDs overlap")
    external = pd.concat([e24_external, qwen], ignore_index=True)
    require(len(external) == 3870 and external["model"].nunique() == 18,
            "Expected 18-class E29 external sample")

    official = load_and_validate(TRAIN)
    dev, _, e0, folds, _ = load_e0_split(official, E0_DIR, TRAIN)
    e24_saved = load_scores(E24_FIT / "decision_scores.csv")
    e0c_saved = load_scores(E0C_SCORES)
    require(len(dev) == 2160 and dev["prompt_id"].nunique() == 120,
            "Official development split changed")
    for key in KEY_COLUMNS:
        require(e24_saved[key].equals(e0c_saved[key]) and e24_saved[key].equals(e0[key]),
                f"Saved E24/E0c/E0 keys differ on {key}")

    official_text = dev["text"].to_numpy()
    official_label = dev["model"].to_numpy()
    e24_text = e24_external["text"].to_numpy()
    e24_label = e24_external["model"].to_numpy()
    e24_weight = e24_external["sample_weight"].to_numpy(dtype=np.float64)
    new_text = external["text"].to_numpy()
    new_label = external["model"].to_numpy()
    new_weight = external["sample_weight"].to_numpy(dtype=np.float64)
    saved_matrix = e24_saved.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float32)
    new_matrix = np.full_like(saved_matrix, np.nan)
    old_max_abs = 0.0
    old_equal_predictions = True
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        require(len(train_idx) == 1728 and len(valid_idx) == 432, "Fold size changed")
        require(not (set(dev.loc[train_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"])),
                "Official prompt overlaps outer fold")
        require(set(official_label[train_idx]) == set(LABELS), "A training class is missing")

        old_model = make_pipeline()
        old_model.fit(
            np.concatenate([official_text[train_idx], e24_text]),
            np.concatenate([official_label[train_idx], e24_label]),
            classifier__sample_weight=np.concatenate([
                np.ones(len(train_idx), dtype=np.float64), e24_weight,
            ]),
        )
        old_score = ordered_scores(old_model, official_text[valid_idx])
        old_diff = float(np.max(np.abs(old_score - saved_matrix[valid_idx])))
        old_max_abs = max(old_max_abs, old_diff)
        old_equal_predictions = old_equal_predictions and np.array_equal(
            old_score.argmax(axis=1), saved_matrix[valid_idx].argmax(axis=1),
        )
        require(old_diff <= 1e-5 and old_equal_predictions,
                f"E24 score reproduction failed at fold {fold}; max abs {old_diff}")

        model = make_pipeline()
        model.fit(
            np.concatenate([official_text[train_idx], new_text]),
            np.concatenate([official_label[train_idx], new_label]),
            classifier__sample_weight=np.concatenate([
                np.ones(len(train_idx), dtype=np.float64), new_weight,
            ]),
        )
        new_matrix[valid_idx] = ordered_scores(model, official_text[valid_idx])
        truth = official_label[valid_idx]
        old_pred = np.asarray(LABELS)[saved_matrix[valid_idx].argmax(axis=1)]
        new_pred = np.asarray(LABELS)[new_matrix[valid_idx].argmax(axis=1)]
        fold_rows.append({
            "fold": fold, "official_train_rows": len(train_idx), "official_valid_rows": len(valid_idx),
            "e24_external_rows": len(e24_external), "e29_external_rows": len(external),
            "e24_e0c_ba": float(balanced_accuracy_score(truth, old_pred)),
            "e29_e0c_ba": float(balanced_accuracy_score(truth, new_pred)),
            "e24_reproduction_max_abs_score_diff": old_diff,
        })
        print(f"Fold {fold}: E24={fold_rows[-1]['e24_e0c_ba']:.6f}, "
              f"E29={fold_rows[-1]['e29_e0c_ba']:.6f}, "
              f"E24 score diff={old_diff:.8g}", flush=True)
    require(np.isfinite(new_matrix).all(), "E29 OOF scores incomplete")
    require(old_equal_predictions, "E24 predictions not exactly reproduced")

    output.mkdir(parents=True, exist_ok=True)
    scores = e24_saved.loc[:, list(KEY_COLUMNS)].copy()
    for position, label in enumerate(LABELS):
        scores[f"score_{label}"] = new_matrix[:, position]
    scores.to_csv(output / "decision_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    external[["id", "prompt_id", "model", "served_model_id", "served_provider",
              "provider_route", "sample_weight"]].to_csv(output / "selected_external.csv", index=False)
    metadata = {
        "experiment": "e29_e24_plus_qwen0902_sensitivity",
        "python_version": sys.version.split()[0], "sklearn_version": sklearn.__version__,
        "official_train_sha256": sha256(TRAIN),
        "e24_saved_scores_sha256": sha256(E24_FIT / "decision_scores.csv"),
        "e24_selection": e24_selection,
        "qwen_selection": qwen_meta,
        "e24_reproduced_max_abs_score_diff": old_max_abs,
        "e24_predictions_exactly_reproduced": old_equal_predictions,
        "n_official_development_rows": len(dev),
        "n_official_development_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": N_FOLDS,
        "n_external_rows": len(external),
        "source_screening_columns_read": False,
        "contest_target_snapshot_proven": False,
        "holdout_evaluated": False,
    }
    (output / "fit_metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"e24_max_abs_score_diff": old_max_abs,
                      "n_qwen_selected": len(qwen), "n_external": len(external)}, indent=2))


def analyze(fit_dir: Path, output: Path) -> None:
    require(not output.exists() or not any(output.iterdir()), "E29 analysis output is not empty")
    fit_meta = json.loads((fit_dir / "fit_metrics.json").read_text(encoding="utf-8"))
    require(fit_meta["e24_predictions_exactly_reproduced"]
            and fit_meta["e24_reproduced_max_abs_score_diff"] <= 1e-5,
            "E24 baseline was not reproduced")
    e24 = load_scores(E24_FIT / "decision_scores.csv")
    e29 = load_scores(fit_dir / "decision_scores.csv")
    roberta = load_scores(ROBERTA_SCORES)
    reference_config = json.loads((SEED_DIRS[42] / "run_config.json").read_text(encoding="utf-8"))
    modern = {seed: validate_run(seed, SEED_DIRS[seed], reference_config)
              for seed in (42, 17, 73)}
    for name, score in [("E29", e29), ("RoBERTa", roberta),
                        *[(f"ModernBERT {seed}", modern[seed]) for seed in (42, 17, 73)]]:
        validate_keys(e24, score, name)
    require(len(e24) == 2160 and e24["prompt_id"].nunique() == 120,
            "Fixed OOF key set changed")
    true = e24["true_model"].to_numpy()
    prompts = e24["prompt_id"].to_numpy()
    labels = np.asarray(LABELS)
    old_score = score_columns(e24)
    new_score = score_columns(e29)
    mean_modern = np.mean(np.stack([
        row_standardize(score_columns(modern[seed])) for seed in (42, 17, 73)
    ]), axis=0)
    zroberta = row_standardize(score_columns(roberta))
    predictions = {
        "e24_e0c": labels[old_score.argmax(axis=1)],
        "e29_e0c": labels[new_score.argmax(axis=1)],
        "e24_fusion": labels[(row_standardize(old_score) + zroberta + mean_modern).argmax(axis=1)],
        "e29_fusion": labels[(row_standardize(new_score) + zroberta + mean_modern).argmax(axis=1)],
    }
    saved = pd.read_csv(E24_ANALYSIS / "oof_predictions.csv", dtype=str, keep_default_na=False)
    validate_keys(e24, saved, "Saved E24 fusion")
    require(np.array_equal(predictions["e24_e0c"], saved["pred_e24_e0c"].to_numpy())
            and np.array_equal(predictions["e24_fusion"], saved["pred_e16_with_e24"].to_numpy())
            and np.isclose(balanced_accuracy_score(true, predictions["e24_fusion"]), 0.425, atol=1e-12),
            "Saved E24 predictions or BA failed to reproduce")

    overall = {name: float(balanced_accuracy_score(true, pred)) for name, pred in predictions.items()}
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        mask = e24["fold"].eq(str(fold)).to_numpy()
        row = {"fold": fold, **{name: float(balanced_accuracy_score(true[mask], pred[mask]))
                                for name, pred in predictions.items()}}
        row["e0c_delta"] = row["e29_e0c"] - row["e24_e0c"]
        row["fusion_delta"] = row["e29_fusion"] - row["e24_fusion"]
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    class_rows = []
    for label in LABELS:
        mask = true == label
        row = {"model": label, "support": int(mask.sum()),
               **{name: float((pred[mask] == label).mean()) for name, pred in predictions.items()}}
        row["e0c_delta"] = row["e29_e0c"] - row["e24_e0c"]
        row["fusion_delta"] = row["e29_fusion"] - row["e24_fusion"]
        class_rows.append(row)
    classes = pd.DataFrame(class_rows)
    require(classes["support"].eq(120).all(), "Expected 120 development rows per class")
    delta = overall["e29_fusion"] - overall["e24_fusion"]
    gate = {
        "fusion_delta_at_least_0_01": delta >= 0.01 - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": int((folds["fusion_delta"] >= -1e-12).sum()) >= 4,
        "no_class_recall_drop_at_least_0_05": float(classes["fusion_delta"].min()) > -0.05 + 1e-12,
    }
    report = {
        "experiment": "e29_e24_plus_qwen0902_sensitivity",
        "fit_metrics_sha256": sha256(fit_dir / "fit_metrics.json"),
        "e29_scores_sha256": sha256(fit_dir / "decision_scores.csv"),
        "e24_scores_sha256": sha256(E24_FIT / "decision_scores.csv"),
        "e24_saved_predictions_exactly_reproduced": True,
        "score": overall,
        "paired_prompt_bootstrap": {
            "e0c_delta": paired_prompt_bootstrap(true, prompts, predictions["e24_e0c"], predictions["e29_e0c"]),
            "fusion_delta": paired_prompt_bootstrap(true, prompts, predictions["e24_fusion"], predictions["e29_fusion"]),
        },
        "fusion_wrong_to_right": int(((predictions["e24_fusion"] != true)
                                      & (predictions["e29_fusion"] == true)).sum()),
        "fusion_right_to_wrong": int(((predictions["e24_fusion"] == true)
                                      & (predictions["e29_fusion"] != true)).sum()),
        "fusion_fold_non_decrease": int((folds["fusion_delta"] >= -1e-12).sum()),
        "worst_fusion_class_recall_delta": float(classes["fusion_delta"].min()),
        "gate": gate,
        "gate_passed": bool(all(gate.values())),
        "weights_tuned": False,
        "contest_target_snapshot_proven": False,
        "holdout_evaluated": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    folds.to_csv(output / "fold_metrics.csv", index=False)
    classes.to_csv(output / "class_metrics.csv", index=False)
    oof = e24.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in predictions.items():
        oof["pred_" + name] = pred
    oof.to_csv(output / "oof_predictions.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "analyze"])
    parser.add_argument("--fit-dir", type=Path, default=Path("outputs/e29_qwen_fit"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("outputs/e29_qwen_analysis"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.fit_dir)
    else:
        analyze(args.fit_dir, args.analysis_dir)


if __name__ == "__main__":
    main()
