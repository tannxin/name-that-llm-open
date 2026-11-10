#!/usr/bin/env python3
"""E24: audited Prvsiyan v9 raw external data in unchanged E0c/E16 folds.

The audit manifest must be locked before `fit`. The script never uses the
source's competition-derived screening columns or official validation rows
for external selection. Run `fit` with original Python/sklearn on the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score

from run_combined import make_pipeline
from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_fixed_blend import KEY_COLUMNS, SCORE_COLUMNS, load_scores
from run_style import load_e0_split


EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_SOURCE_SHA256 = "393f300c7b521507a55f5a8793bd2256519ed6324da97153f8cd2b3503f01ce0"
SOURCE_COLUMNS = (
    "id", "prompt_id", "model", "text", "split", "served_model_id",
    "served_provider", "provider_route", "call_id",
)
AUDIT_KEYS = (
    "id", "prompt_id", "model", "call_id", "served_model_id",
    "served_provider", "provider_route",
)
PROMPT_HASH_PREFIX = "name-that-llm:e24:v1:seed42:"
MAX_PROMPTS = 250
EXTERNAL_WEIGHT_PER_CLASS = 24.0


def fixed_prompt_hash(prompt_id: str) -> str:
    return hashlib.sha256((PROMPT_HASH_PREFIX + prompt_id).encode("utf-8")).hexdigest()


def select_external(source_path: Path, audit_path: Path, report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select only audited raw rows, without opening any official data."""
    if sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Raw Prvsiyan v9 passages.csv SHA-256 differs from pre-registration")
    if not audit_path.is_file():
        raise ValueError("Locked accepted audit manifest is missing; no source audit, no fit")
    if not report_path.is_file():
        raise ValueError("Completed provenance audit report is missing; no audit, no fit")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("audit_passed") is not True or report.get("source_version") != 9
            or report.get("source_file_sha256", {}).get("passages.csv") != EXPECTED_SOURCE_SHA256
            or report.get("accepted_manifest_sha256") != sha256(audit_path)
            or report.get("envelopes", {}).get("new_calls_matched") != 5400
            or report.get("envelopes", {}).get("mismatched_fields") != 0
            or report.get("envelopes", {}).get("missing_calls") != 0
            or report.get("competition_train_derived_source_columns_used") is not False):
        raise ValueError("Locked v9 provenance report does not certify the required audit")
    source = pd.read_csv(
        source_path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
        usecols=list(SOURCE_COLUMNS),
    )
    if source.empty or source["id"].duplicated().any():
        raise ValueError("Raw source is empty or contains duplicate row IDs")
    raw_new = source.loc[source["split"].eq("new")]
    all_prompts = sorted(raw_new["prompt_id"].unique())
    if len(all_prompts) != 300 or len(raw_new) != 5400:
        raise ValueError("Raw v9 split=new differs from the pre-registered 300-prompt corpus")
    prompt_order = sorted(all_prompts, key=lambda value: (fixed_prompt_hash(value), value))
    chosen_prompts = set(prompt_order[:MAX_PROMPTS])
    prompt_manifest = pd.DataFrame({
        "prompt_id": prompt_order,
        "hash": [fixed_prompt_hash(value) for value in prompt_order],
        "selected": [value in chosen_prompts for value in prompt_order],
    })

    audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(AUDIT_KEYS) - set(audit.columns))
    if missing or audit.empty or audit["id"].duplicated().any():
        raise ValueError(f"Audit manifest missing keys {missing}, empty, or has duplicate IDs")
    if ("source_file_sha256" not in audit or
            not audit["source_file_sha256"].eq(EXPECTED_SOURCE_SHA256).all() or
            len(audit) != report.get("accepted_rows")):
        raise ValueError("Audit manifest source hash or accepted row count differs from report")
    if audit.loc[:, list(AUDIT_KEYS)].eq("").any().any():
        raise ValueError("Audit manifest has blank provenance keys")
    if not set(audit["id"]).issubset(set(raw_new["id"])):
        raise ValueError("Audit manifest contains IDs outside raw split=new")
    audited = source.set_index("id", drop=False).loc[audit["id"].to_numpy()].reset_index(drop=True)
    for key in AUDIT_KEYS:
        if not np.array_equal(audited[key].to_numpy(), audit[key].to_numpy()):
            raise ValueError(f"Audited {key} does not match raw v9 source")
    if not audited["split"].eq("new").all():
        raise ValueError("Audit manifest includes an unregistered source split")
    if not set(audited["model"]).issubset(set(LABELS) - {"qwen3.8-max"}):
        raise ValueError("Audit manifest includes unknown or unverified Qwen target labels")
    if audited["provider_route"].eq("kaggle_proxy").any():
        raise ValueError("Audit manifest includes Kaggle proxy rows without proven backend")
    allowed_routes = {"anthropic", "openai", "google", "openrouter"}
    if not set(audited["provider_route"]).issubset(allowed_routes):
        raise ValueError("Audit manifest includes an unverified route")
    expected_route = {
        "claude": "anthropic", "gpt": "openai", "gemini": "google",
        "glm": "openrouter", "kimi": "openrouter",
    }
    for prefix, route in expected_route.items():
        mask = audited["model"].str.startswith(prefix + "-")
        if not audited.loc[mask, "provider_route"].eq(route).all():
            raise ValueError(f"Audit manifest has a non-whitelisted {prefix} route")
    if audited["served_provider"].eq("").any() or audited["text"].str.strip().eq("").any():
        raise ValueError("Audited rows lack served provider or response text")

    selected = audited.loc[audited["prompt_id"].isin(chosen_prompts)].copy()
    if selected.duplicated(["prompt_id", "model"]).any():
        raise ValueError("Accepted source has multiple responses for one external prompt/class")
    if selected.empty or len(selected) > MAX_PROMPTS * (len(LABELS) - 1):
        raise ValueError("No eligible external rows or more than 250 per supported class")
    selected = selected.sort_values(["prompt_id", "model", "id"], kind="stable").reset_index(drop=True)
    counts = selected["model"].value_counts().to_dict()
    if any(count > MAX_PROMPTS for count in counts.values()):
        raise ValueError("External sample limit exceeded")
    selected["sample_weight"] = [EXTERNAL_WEIGHT_PER_CLASS / counts[label]
                                 for label in selected["model"]]
    effective = selected.groupby("model")["sample_weight"].sum()
    if not np.allclose(effective.to_numpy(), EXTERNAL_WEIGHT_PER_CLASS, atol=1e-12):
        raise AssertionError("External class weight does not total 24")
    selection_meta = {
        "dataset": "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
        "dataset_version": 9,
        "raw_source_sha256": EXPECTED_SOURCE_SHA256,
        "audit_manifest_sha256": sha256(audit_path),
        "audit_report_sha256": sha256(report_path),
        "selection_rule": "sort all 300 raw split=new prompt_id by SHA256(prefix+prompt_id); take first 250; accept audited rows within them; no backfill",
        "prompt_hash_prefix": PROMPT_HASH_PREFIX,
        "n_raw_new_prompts": len(all_prompts),
        "n_selected_prompt_ids": MAX_PROMPTS,
        "n_accepted_rows_before_prompt_selection": len(audited),
        "n_selected_external_rows": len(selected),
        "selected_rows_by_class": {label: int(counts.get(label, 0)) for label in LABELS},
        "selected_rows_by_route": {key: int(value) for key, value in selected["provider_route"].value_counts().items()},
        "external_total_weight_per_covered_class": EXTERNAL_WEIGHT_PER_CLASS,
    }
    return selected, prompt_manifest, selection_meta


def ordered_scores(model, texts: np.ndarray) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_
    if set(classes) != set(LABELS):
        raise ValueError("E0c fit lost an official target class")
    positions = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    return model.decision_function(texts)[:, positions].astype(np.float32)


def fit(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    if sklearn.__version__ != "1.7.2" or sys.version_info[:3] != (3, 10, 12):
        raise ValueError("Fit requires original Python 3.10.12 / scikit-learn 1.7.2 environment")
    external, prompt_manifest, selection_meta = select_external(
        args.source, args.audit_manifest, args.audit_report)

    if sha256(args.train) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Official train.csv differs from frozen E0/E16 experiment")
    official = load_and_validate(args.train)
    dev, _, e0, folds, _ = load_e0_split(official, args.e0_dir, args.train)
    baseline = load_scores(args.e0c_scores)
    saved_pred = pd.read_csv(args.e0c_predictions, dtype=str, keep_default_na=False)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Expected original 120-prompt/2,160-row development partition")
    for key in KEY_COLUMNS:
        if not baseline[key].equals(saved_pred[key]) or not baseline[key].equals(e0[key]):
            raise ValueError(f"Saved E0c/E0 files differ on {key}")
    base_margin = baseline.loc[:, list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    base_prediction = np.asarray(LABELS)[base_margin.argmax(axis=1)]
    if not np.array_equal(base_prediction, saved_pred["pred_model"].to_numpy()):
        raise ValueError("Original E0c margins do not reproduce saved predictions")
    if not np.isclose(balanced_accuracy_score(dev["model"], base_prediction),
                      0.34259259259259256, atol=1e-12):
        raise ValueError("Original E0c baseline BA failed to reproduce")

    official_text = dev["text"].to_numpy()
    official_label = dev["model"].to_numpy()
    external_text = external["text"].to_numpy()
    external_label = external["model"].to_numpy()
    external_weight = external["sample_weight"].to_numpy(dtype=np.float64)
    selected_score = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    selected_prediction = np.empty(len(dev), dtype=object)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(dev.loc[train_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Official prompt groups overlap in fold {fold}")
        if set(official_label[train_idx]) != set(LABELS) or set(official_label[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} lacks a target class")
        if len(train_idx) != 1728 or len(valid_idx) != 432:
            raise ValueError(f"Fold {fold} has unexpected official row counts")
        texts = np.concatenate((official_text[train_idx], external_text))
        labels = np.concatenate((official_label[train_idx], external_label))
        weights = np.concatenate((np.ones(len(train_idx), dtype=np.float64), external_weight))
        model = make_pipeline()
        model.fit(texts, labels, classifier__sample_weight=weights)
        selected_score[valid_idx] = ordered_scores(model, official_text[valid_idx])
        selected_prediction[valid_idx] = np.asarray(LABELS)[selected_score[valid_idx].argmax(axis=1)]
        fold_rows.append({
            "fold": fold,
            "official_train_rows": len(train_idx),
            "official_valid_rows": len(valid_idx),
            "external_train_rows": len(external),
            "original_e0c_ba": float(balanced_accuracy_score(official_label[valid_idx], base_prediction[valid_idx])),
            "e24_e0c_ba": float(balanced_accuracy_score(official_label[valid_idx], selected_prediction[valid_idx])),
        })
        print(f"Fold {fold}/{N_FOLDS}: E0c={fold_rows[-1]['original_e0c_ba']:.6f}, "
              f"E24={fold_rows[-1]['e24_e0c_ba']:.6f}", flush=True)
    if not np.isfinite(selected_score).all():
        raise ValueError("E24 OOF decision scores are incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = baseline.loc[:, list(KEY_COLUMNS)].copy()
    for index, label in enumerate(LABELS):
        scores[f"score_{label}"] = selected_score[:, index]
    scores.to_csv(args.output_dir / "decision_scores.csv", index=False)
    oof = baseline.loc[:, list(KEY_COLUMNS)].copy()
    oof["pred_model"] = selected_prediction
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    external.loc[:, ["id", "prompt_id", "model", "served_model_id", "served_provider",
                     "provider_route", "sample_weight"]].to_csv(
                         args.output_dir / "selected_external.csv", index=False)
    prompt_manifest.to_csv(args.output_dir / "selected_prompt_ids.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    metadata = {
        "experiment": "e24_prvsiyan_raw_external_ablation",
        "stage": "fit",
        "official_train_sha256": sha256(args.train),
        "original_e0c_scores_sha256": sha256(args.e0c_scores),
        "original_e0c_predictions_sha256": sha256(args.e0c_predictions),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "selection": selection_meta,
        "n_official_development_rows": len(dev),
        "n_official_development_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": N_FOLDS,
        "original_e0c_ba": float(balanced_accuracy_score(official_label, base_prediction)),
        "e24_e0c_ba": float(balanced_accuracy_score(official_label, selected_prediction)),
        "official_training_weight_per_class_per_fold": 96.0,
        "external_weight_per_covered_class_per_fold": EXTERNAL_WEIGHT_PER_CLASS,
        "baseline_predictions_reproduced": True,
        "source_screening_columns_read": False,
        "exploratory_due_to_source_protocol_using_official_global_length_statistics": True,
    }
    (args.output_dir / "fit_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"e0c_ba": metadata["original_e0c_ba"],
                      "e24_e0c_ba": metadata["e24_e0c_ba"],
                      "selected_external_rows": len(external)}, indent=2), flush=True)


def analyze(args: argparse.Namespace) -> None:
    from analyze_modernbert_seeds import score_columns, validate_keys, validate_run
    from run_score_normalization import paired_prompt_bootstrap, row_standardize

    fit_meta = json.loads((args.fit_dir / "fit_metrics.json").read_text(encoding="utf-8"))
    if not fit_meta["baseline_predictions_reproduced"]:
        raise ValueError("Original E0c baseline not verified")
    original = load_scores(args.e0c_scores)
    e24 = load_scores(args.fit_dir / "decision_scores.csv")
    roberta = load_scores(args.roberta_scores)
    reference_config = json.loads((args.seed42_dir / "run_config.json").read_text(encoding="utf-8"))
    seed_dirs = {42: args.seed42_dir, 17: args.seed17_dir, 73: args.seed73_dir}
    modern = {seed: validate_run(seed, seed_dirs[seed], reference_config)
              for seed in (42, 17, 73)}
    for name, data in [("E24", e24), ("RoBERTa", roberta),
                       *[(f"ModernBERT seed {seed}", data) for seed, data in modern.items()]]:
        validate_keys(original, data, name)
    saved_e16 = pd.read_csv(args.e16_oof, dtype=str, keep_default_na=False)
    saved_e24 = pd.read_csv(args.fit_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    validate_keys(original, saved_e16, "Saved E16")
    validate_keys(original, saved_e24, "Saved E24")
    if len(original) != 2160 or original["prompt_id"].nunique() != 120:
        raise ValueError("Expected original 2,160-row, 120-prompt OOF")

    labels = np.asarray(LABELS)
    true, prompts = original["true_model"].to_numpy(), original["prompt_id"].to_numpy()
    old_score, new_score = score_columns(original), score_columns(e24)
    base_pred, e24_pred = labels[old_score.argmax(axis=1)], labels[new_score.argmax(axis=1)]
    if not np.array_equal(e24_pred, saved_e24["pred_model"].to_numpy()):
        raise ValueError("E24 score argmax differs from fit predictions")
    if not np.array_equal(base_pred, pd.read_csv(args.e0c_predictions, dtype=str)["pred_model"].to_numpy()):
        raise ValueError("Original E0c score argmax differs from saved predictions")
    mean_modern = np.mean(np.stack([row_standardize(score_columns(modern[seed]))
                                    for seed in (42, 17, 73)]), axis=0)
    zroberta = row_standardize(score_columns(roberta))
    old_fusion = labels[(row_standardize(old_score) + zroberta + mean_modern).argmax(axis=1)]
    new_fusion = labels[(row_standardize(new_score) + zroberta + mean_modern).argmax(axis=1)]
    if not np.array_equal(old_fusion, saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("Original E16 fixed fusion did not reproduce")
    if not np.isclose(balanced_accuracy_score(true, old_fusion), 0.40925925925925927, atol=1e-12):
        raise ValueError("Original E16 fixed fusion BA did not reproduce")

    predictions = {"e0c": base_pred, "e24_e0c": e24_pred,
                   "e16": old_fusion, "e16_with_e24": new_fusion}
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        mask = original["fold"].eq(str(fold)).to_numpy()
        row = {"fold": fold, **{name: float(balanced_accuracy_score(true[mask], pred[mask]))
                                for name, pred in predictions.items()}}
        row["e0c_delta"] = row["e24_e0c"] - row["e0c"]
        row["fusion_delta"] = row["e16_with_e24"] - row["e16"]
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    class_rows = []
    for label in LABELS:
        mask = true == label
        row = {"model": label, "support": int(mask.sum()),
               **{name: float((pred[mask] == label).mean()) for name, pred in predictions.items()}}
        row["e0c_delta"] = row["e24_e0c"] - row["e0c"]
        row["fusion_delta"] = row["e16_with_e24"] - row["e16"]
        class_rows.append(row)
    class_metrics = pd.DataFrame(class_rows)
    if not (class_metrics["support"] == 120).all():
        raise ValueError("Expected exactly 120 official development rows per class")
    fusion_delta = float(balanced_accuracy_score(true, new_fusion) -
                         balanced_accuracy_score(true, old_fusion))
    gate = {
        "fusion_delta_at_least_0_01": fusion_delta >= 0.01 - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": int((fold_metrics["fusion_delta"] >= -1e-12).sum()) >= 4,
        "no_class_recall_drop_at_least_0_05": float(class_metrics["fusion_delta"].min()) > -0.05 + 1e-12,
    }
    metrics = {
        "experiment": "e24_prvsiyan_raw_external_ablation",
        "stage": "analysis",
        "fit_metrics_sha256": sha256(args.fit_dir / "fit_metrics.json"),
        "e24_scores_sha256": sha256(args.fit_dir / "decision_scores.csv"),
        "original_e0c_scores_sha256": sha256(args.e0c_scores),
        "roberta_scores_sha256": sha256(args.roberta_scores),
        "modern_seed_score_sha256": {str(seed): sha256(seed_dirs[seed] / "oof_predictions.csv")
                                     for seed in (42, 17, 73)},
        "e16_saved_oof_sha256": sha256(args.e16_oof),
        "saved_e16_predictions_exactly_reproduced": True,
        "score": {name: float(balanced_accuracy_score(true, pred))
                  for name, pred in predictions.items()},
        "paired_prompt_bootstrap": {
            "e0c_delta": paired_prompt_bootstrap(true, prompts, base_pred, e24_pred),
            "fusion_delta": paired_prompt_bootstrap(true, prompts, old_fusion, new_fusion),
        },
        "fusion_wrong_to_right": int(((old_fusion != true) & (new_fusion == true)).sum()),
        "fusion_right_to_wrong": int(((old_fusion == true) & (new_fusion != true)).sum()),
        "fusion_fold_non_decrease": int((fold_metrics["fusion_delta"] >= -1e-12).sum()),
        "worst_fusion_class_recall_delta": float(class_metrics["fusion_delta"].min()),
        "gate": gate,
        "gate_passed": bool(all(gate.values())),
        "fixed_fusion_rule": "row-z E0c or E24 + row-z RoBERTa + mean of three row-z ModernBERT seeds",
        "weights_tuned": False,
        "exploratory_due_to_source_protocol_using_official_global_length_statistics": True,
        "holdout_evaluated": False,
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    oof = original.loc[:, list(KEY_COLUMNS)].copy()
    for name, pred in predictions.items():
        oof[f"pred_{name}"] = pred
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fitting = stages.add_parser("fit")
    fitting.add_argument("--source", type=Path, default=Path("data/external/prvsiyan/passages.csv"))
    fitting.add_argument("--audit-manifest", type=Path,
                         default=Path("data/external/prvsiyan/e24_accepted_manifest.csv"))
    fitting.add_argument("--audit-report", type=Path,
                         default=Path("outputs/e24_prvsiyan_audit/report.json"))
    fitting.add_argument("--train", type=Path, default=Path("data/train.csv"))
    fitting.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    fitting.add_argument("--e0c-predictions", type=Path,
                         default=Path("outputs/e0c_tfidf_plus_style/oof_predictions.csv"))
    fitting.add_argument("--e0c-scores", type=Path,
                         default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    fitting.add_argument("--output-dir", type=Path, default=Path("outputs/e24_prvsiyan_fit"))
    analysis = stages.add_parser("analyze")
    analysis.add_argument("--fit-dir", type=Path, default=Path("outputs/e24_prvsiyan_fit"))
    analysis.add_argument("--e0c-predictions", type=Path,
                          default=Path("outputs/e0c_tfidf_plus_style/oof_predictions.csv"))
    analysis.add_argument("--e0c-scores", type=Path,
                          default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    analysis.add_argument("--roberta-scores", type=Path,
                          default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    analysis.add_argument("--seed42-dir", type=Path, default=Path("outputs/e15_modernbert_base_12ep_eager"))
    analysis.add_argument("--seed17-dir", type=Path, default=Path("outputs/e16_modernbert_seed17_eager"))
    analysis.add_argument("--seed73-dir", type=Path, default=Path("outputs/e16_modernbert_seed73_eager"))
    analysis.add_argument("--e16-oof", type=Path,
                          default=Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv"))
    analysis.add_argument("--output-dir", type=Path, default=Path("outputs/e24_prvsiyan_analysis"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
