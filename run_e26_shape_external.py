#!/usr/bin/env python3
"""E26: E20's locked shape block on E24's audited external rows and folds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS, N_FOLDS, load_and_validate, sha256
from run_e20_shape_ablation import make_shape_pipeline, shape_texts
from run_e24_prvsiyan_external_ablation import (
    EXPECTED_TRAIN_SHA256, ordered_scores, select_external,
)
from run_fixed_blend import KEY_COLUMNS, load_scores
from run_style import load_e0_split


E24_SCORE_SHA256 = "845c657f03b761f7b8f62fc17a2da583b6ecfd5962d5be1ae6fe485e5a9c3575"
EXTERNAL_COLUMNS = [
    "id", "prompt_id", "model", "served_model_id", "served_provider",
    "provider_route", "sample_weight",
]


def verify_external_selection(selected: pd.DataFrame, saved_path: Path) -> None:
    saved = pd.read_csv(saved_path, dtype={col: str for col in EXTERNAL_COLUMNS[:-1]},
                        keep_default_na=False)
    if len(selected) != len(saved) or len(selected) != 3620:
        raise ValueError("E26 external selection differs in length from locked E24")
    for col in EXTERNAL_COLUMNS[:-1]:
        if not np.array_equal(selected[col].to_numpy(), saved[col].to_numpy()):
            raise ValueError(f"E26 external selection differs from E24 on {col}")
    if not np.allclose(selected["sample_weight"].to_numpy(dtype=float),
                       saved["sample_weight"].to_numpy(dtype=float), rtol=0, atol=1e-14):
        raise ValueError("E26 external sample weights differ from locked E24")


def fit(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    if sys.version_info[:3] != (3, 10, 12) or sklearn.__version__ != "1.7.2":
        raise ValueError("E26 fitting requires original Python 3.10.12 / sklearn 1.7.2")
    if shape_texts(np.array(["Hello WORLD x3 _keep.\nAa"])) != ["x X x3 _x.\nx"]:
        raise AssertionError("Shape transform changed from E20")
    if sha256(args.train) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Official training file differs from E24")
    if sha256(args.e24_scores) != E24_SCORE_SHA256:
        raise ValueError("Locked E24 baseline score file changed")

    started = time.perf_counter()
    external, prompt_manifest, selection_meta = select_external(
        args.source, args.audit_manifest, args.audit_report)
    verify_external_selection(external, args.e24_selection)
    official = load_and_validate(args.train)
    dev, _, e0, folds, _ = load_e0_split(official, args.e0_dir, args.train)
    baseline = load_scores(args.e24_scores)
    saved_e24 = pd.read_csv(args.e24_predictions, dtype=str, keep_default_na=False)
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Original development partition changed")
    for col in KEY_COLUMNS:
        if not baseline[col].equals(saved_e24[col]) or not baseline[col].equals(e0[col]):
            raise ValueError(f"E24 baseline row alignment failed on {col}")
    labels = np.asarray(LABELS)
    baseline_score = baseline[[f"score_{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    baseline_pred = labels[baseline_score.argmax(axis=1)]
    if not np.array_equal(baseline_pred, saved_e24["pred_model"].to_numpy()):
        raise ValueError("E24 baseline predictions do not match locked OOF")
    if not np.isclose(balanced_accuracy_score(dev["model"], baseline_pred),
                      0.3726851851851852, atol=1e-12):
        raise ValueError("E24 single-stream baseline BA changed")

    official_text = dev["text"].to_numpy()
    official_label = dev["model"].to_numpy()
    external_text = external["text"].to_numpy()
    external_label = external["model"].to_numpy()
    external_weight = external["sample_weight"].to_numpy(dtype=np.float64)
    scores = np.full((len(dev), len(LABELS)), np.nan, dtype=np.float32)
    predictions = np.empty(len(dev), dtype=object)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if (len(train_idx), len(valid_idx)) != (1728, 432):
            raise ValueError(f"Official row counts changed in fold {fold}")
        if set(dev.loc[train_idx, "prompt_id"]) & set(dev.loc[valid_idx, "prompt_id"]):
            raise ValueError(f"Official prompt leakage in fold {fold}")
        texts = np.concatenate((official_text[train_idx], external_text))
        targets = np.concatenate((official_label[train_idx], external_label))
        weights = np.concatenate((np.ones(len(train_idx), dtype=np.float64), external_weight))
        fold_started = time.perf_counter()
        model = make_shape_pipeline()
        model.fit(texts, targets, classifier__sample_weight=weights)
        scores[valid_idx] = ordered_scores(model, official_text[valid_idx])
        predictions[valid_idx] = labels[scores[valid_idx].argmax(axis=1)]
        elapsed = time.perf_counter() - fold_started
        fold_rows.append({
            "fold": fold,
            "seconds": elapsed,
            "official_train_rows": len(train_idx),
            "official_valid_rows": len(valid_idx),
            "external_train_rows": len(external),
            "e24_single_ba": float(balanced_accuracy_score(
                official_label[valid_idx], baseline_pred[valid_idx])),
            "e26_single_ba": float(balanced_accuracy_score(
                official_label[valid_idx], predictions[valid_idx])),
        })
        print(f"fold {fold}/{N_FOLDS}: {elapsed:.1f}s, "
              f"E24={fold_rows[-1]['e24_single_ba']:.6f}, "
              f"E26={fold_rows[-1]['e26_single_ba']:.6f}", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("E26 OOF scores are incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_frame = baseline.loc[:, list(KEY_COLUMNS)].copy()
    for index, label in enumerate(LABELS):
        score_frame[f"score_{label}"] = scores[:, index]
    score_frame.to_csv(args.output_dir / "decision_scores.csv", index=False)
    oof = baseline.loc[:, list(KEY_COLUMNS)].copy()
    oof["pred_model"] = predictions
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    external.loc[:, EXTERNAL_COLUMNS].to_csv(args.output_dir / "selected_external.csv", index=False)
    prompt_manifest.to_csv(args.output_dir / "selected_prompt_ids.csv", index=False)
    metadata = {
        "experiment": "e26_shape_on_e24_external",
        "stage": "fit",
        "preregistration": "docs/e26_shape_external.md",
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "official_train_sha256": sha256(args.train),
        "e24_scores_sha256": sha256(args.e24_scores),
        "e24_selection_sha256": sha256(args.e24_selection),
        "external_selection_exactly_reproduced": True,
        "e24_predictions_exactly_reproduced": True,
        "selected_external": selection_meta,
        "shape_spec": "exact E20 make_shape_pipeline, block weight 0.5",
        "e24_single_ba": float(balanced_accuracy_score(official_label, baseline_pred)),
        "e26_single_ba": float(balanced_accuracy_score(official_label, predictions)),
        "cpu_wall_seconds": time.perf_counter() - started,
        "exploratory_due_to_source_protocol_using_official_global_length_statistics": True,
    }
    (args.output_dir / "fit_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"e24_single_ba": metadata["e24_single_ba"],
                      "e26_single_ba": metadata["e26_single_ba"],
                      "cpu_wall_seconds": metadata["cpu_wall_seconds"]}, indent=2), flush=True)


def analyze(args: argparse.Namespace) -> None:
    from analyze_modernbert_seeds import score_columns, validate_keys, validate_run
    from run_score_normalization import paired_prompt_bootstrap, row_standardize

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    fit_meta = json.loads((args.fit_dir / "fit_metrics.json").read_text(encoding="utf-8"))
    if not fit_meta["external_selection_exactly_reproduced"]:
        raise ValueError("E24 external selection was not verified")
    if sha256(args.e24_scores) != E24_SCORE_SHA256:
        raise ValueError("Locked E24 baseline score file changed")
    e0c = load_scores(args.e0c_scores)
    e24 = load_scores(args.e24_scores)
    e26 = load_scores(args.fit_dir / "decision_scores.csv")
    roberta = load_scores(args.roberta_scores)
    seed_dirs = {42: args.seed42_dir, 17: args.seed17_dir, 73: args.seed73_dir}
    config = json.loads((args.seed42_dir / "run_config.json").read_text(encoding="utf-8"))
    modern = {seed: validate_run(seed, seed_dirs[seed], config) for seed in (42, 17, 73)}
    for name, frame in [("E0c", e0c), ("E26", e26), ("RoBERTa", roberta),
                        *[(f"ModernBERT seed {seed}", frame) for seed, frame in modern.items()]]:
        validate_keys(e24, frame, name)
    saved_e26 = pd.read_csv(args.fit_dir / "oof_predictions.csv", dtype=str, keep_default_na=False)
    saved_e24 = pd.read_csv(args.e24_predictions, dtype=str, keep_default_na=False)
    saved_e16 = pd.read_csv(args.e16_fusion, dtype=str, keep_default_na=False)
    saved_fusion = pd.read_csv(args.e24_fusion, dtype=str, keep_default_na=False)
    for name, frame in [("E26 OOF", saved_e26), ("E24 OOF", saved_e24),
                        ("E16 fusion", saved_e16), ("E24 fusion", saved_fusion)]:
        validate_keys(e24, frame, name)
    labels = np.asarray(LABELS)
    true = e24["true_model"].to_numpy()
    prompts = e24["prompt_id"].to_numpy()
    e24_score, e26_score = score_columns(e24), score_columns(e26)
    e24_pred, e26_pred = labels[e24_score.argmax(axis=1)], labels[e26_score.argmax(axis=1)]
    if not np.array_equal(e24_pred, saved_e24["pred_model"].to_numpy()):
        raise ValueError("E24 single-stream OOF failed to reproduce")
    if not np.array_equal(e26_pred, saved_e26["pred_model"].to_numpy()):
        raise ValueError("E26 single-stream OOF failed to reproduce")
    mean_modern = np.mean(np.stack([
        row_standardize(score_columns(modern[seed])) for seed in (42, 17, 73)
    ]), axis=0)
    zroberta = row_standardize(score_columns(roberta))
    e16_fusion = labels[(row_standardize(score_columns(e0c)) + zroberta + mean_modern).argmax(axis=1)]
    e24_fusion = labels[(row_standardize(e24_score) + zroberta + mean_modern).argmax(axis=1)]
    e26_fusion = labels[(row_standardize(e26_score) + zroberta + mean_modern).argmax(axis=1)]
    if not np.array_equal(e16_fusion, saved_e16["pred_e10_modern_three_seed_mean"].to_numpy()):
        raise ValueError("E16 fixed fusion failed exact OOF reproduction")
    if not np.isclose(balanced_accuracy_score(true, e16_fusion),
                      0.40925925925925927, atol=1e-12):
        raise ValueError("E16 fixed fusion BA changed")
    if not np.array_equal(e24_fusion, saved_fusion["pred_e16_with_e24"].to_numpy()):
        raise ValueError("E24 fixed fusion failed exact OOF reproduction")
    if not np.isclose(balanced_accuracy_score(true, e24_fusion), 0.425, atol=1e-12):
        raise ValueError("E24 fixed fusion baseline BA changed")

    predictions = {
        "e24_single": e24_pred, "e26_single": e26_pred,
        "e24_fusion": e24_fusion, "e26_fusion": e26_fusion,
    }
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        mask = e24["fold"].eq(str(fold)).to_numpy()
        row = {"fold": fold, **{name: float(balanced_accuracy_score(true[mask], pred[mask]))
                                for name, pred in predictions.items()}}
        row["single_delta"] = row["e26_single"] - row["e24_single"]
        row["fusion_delta"] = row["e26_fusion"] - row["e24_fusion"]
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    class_rows = []
    for label in LABELS:
        mask = true == label
        row = {"model": label, "support": int(mask.sum()),
               **{name: float((pred[mask] == label).mean()) for name, pred in predictions.items()}}
        row["single_delta"] = row["e26_single"] - row["e24_single"]
        row["fusion_delta"] = row["e26_fusion"] - row["e24_fusion"]
        class_rows.append(row)
    class_metrics = pd.DataFrame(class_rows)
    if not (class_metrics["support"] == 120).all():
        raise ValueError("Expected 120 development rows per class")
    delta = float(balanced_accuracy_score(true, e26_fusion) -
                  balanced_accuracy_score(true, e24_fusion))
    gate = {
        "fusion_delta_at_least_0_01": delta >= 0.01 - 1e-12,
        "at_least_4_of_5_folds_non_decreasing": int((fold_metrics["fusion_delta"] >= -1e-12).sum()) >= 4,
        "no_class_recall_drop_at_least_0_05": float(class_metrics["fusion_delta"].min()) > -0.05 + 1e-12,
    }
    metrics = {
        "experiment": "e26_shape_on_e24_external",
        "stage": "analysis",
        "fit_metrics_sha256": sha256(args.fit_dir / "fit_metrics.json"),
        "e24_scores_sha256": sha256(args.e24_scores),
        "e26_scores_sha256": sha256(args.fit_dir / "decision_scores.csv"),
        "e24_fusion_predictions_exactly_reproduced": True,
        "e16_fusion_predictions_exactly_reproduced": True,
        "score": {name: float(balanced_accuracy_score(true, pred)) for name, pred in predictions.items()},
        "paired_prompt_bootstrap": {
            "single_delta": paired_prompt_bootstrap(true, prompts, e24_pred, e26_pred),
            "fusion_delta": paired_prompt_bootstrap(true, prompts, e24_fusion, e26_fusion),
        },
        "fusion_wrong_to_right": int(((e24_fusion != true) & (e26_fusion == true)).sum()),
        "fusion_right_to_wrong": int(((e24_fusion == true) & (e26_fusion != true)).sum()),
        "fusion_folds_non_decreasing": int((fold_metrics["fusion_delta"] >= -1e-12).sum()),
        "worst_class_recall_delta": float(class_metrics["fusion_delta"].min()),
        "gate": gate,
        "gate_passed": bool(all(gate.values())),
        "fixed_fusion_rule": "row-z E24 or E26 + row-z RoBERTa + mean of three row-z ModernBERT seeds",
        "weights_tuned": False,
        "exploratory_due_to_source_protocol_using_official_global_length_statistics": True,
    }
    if args.e25_b_seed42_oof.is_file():
        e25 = load_scores(args.e25_b_seed42_oof)
        validate_keys(e24, e25, "E25 B seed42")
        mean_modern_e25 = (mean_modern * 3 - row_standardize(score_columns(modern[42]))
                           + row_standardize(score_columns(e25))) / 3
        e24_e25 = labels[(row_standardize(e24_score) + zroberta + mean_modern_e25).argmax(axis=1)]
        e26_e25 = labels[(row_standardize(e26_score) + zroberta + mean_modern_e25).argmax(axis=1)]
        metrics["secondary_fixed_e25_b_seed42"] = {
            "e25_oof_sha256": sha256(args.e25_b_seed42_oof),
            "e24_plus_e25_b_ba": float(balanced_accuracy_score(true, e24_e25)),
            "e26_plus_e25_b_ba": float(balanced_accuracy_score(true, e26_e25)),
            "delta": float(balanced_accuracy_score(true, e26_e25) -
                           balanced_accuracy_score(true, e24_e25)),
            "not_used_for_gate_or_tuning": True,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    oof = e24.loc[:, list(KEY_COLUMNS)].copy()
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
    fitting.add_argument("--e24-scores", type=Path,
                         default=Path("outputs/e24_prvsiyan_fit/decision_scores.csv"))
    fitting.add_argument("--e24-predictions", type=Path,
                         default=Path("outputs/e24_prvsiyan_fit/oof_predictions.csv"))
    fitting.add_argument("--e24-selection", type=Path,
                         default=Path("outputs/e24_prvsiyan_fit/selected_external.csv"))
    fitting.add_argument("--output-dir", type=Path,
                         default=Path("outputs/e26_shape_external_fit"))
    analysis = stages.add_parser("analyze")
    analysis.add_argument("--fit-dir", type=Path, default=Path("outputs/e26_shape_external_fit"))
    analysis.add_argument("--e0c-scores", type=Path,
                          default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    analysis.add_argument("--e24-scores", type=Path,
                          default=Path("outputs/e24_prvsiyan_fit/decision_scores.csv"))
    analysis.add_argument("--e24-predictions", type=Path,
                          default=Path("outputs/e24_prvsiyan_fit/oof_predictions.csv"))
    analysis.add_argument("--e24-fusion", type=Path,
                          default=Path("outputs/e24_prvsiyan_analysis/oof_predictions.csv"))
    analysis.add_argument("--e16-fusion", type=Path,
                          default=Path("outputs/e16_modernbert_seed_analysis_checked/oof_predictions.csv"))
    analysis.add_argument("--roberta-scores", type=Path,
                          default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    analysis.add_argument("--seed42-dir", type=Path,
                          default=Path("outputs/e15_modernbert_base_12ep_eager"))
    analysis.add_argument("--seed17-dir", type=Path,
                          default=Path("outputs/e16_modernbert_seed17_eager"))
    analysis.add_argument("--seed73-dir", type=Path,
                          default=Path("outputs/e16_modernbert_seed73_eager"))
    analysis.add_argument("--e25-b-seed42-oof", type=Path,
                          default=Path("outputs/e25_modernbert_B_seed42/oof_predictions.csv"))
    analysis.add_argument("--output-dir", type=Path,
                          default=Path("outputs/e26_shape_external_analysis"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
