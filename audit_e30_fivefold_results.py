"""Independently audit the locked E30 official-development five-fold OOF.

Small CSV/JSON run artifacts must first be copied into the Git-ignored snapshot
directory below. This script reads neither the official training CSV nor any
holdout/test responses; its only tracked output is an aggregate JSON report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "outputs/e30_qwen3_embedding8b_audit_snapshot"
OUT = ROOT / "docs/e30_embedding8b_fivefold_audit_aggregate.json"
KEYS = ["id", "prompt_id", "true_model", "fold"]
OFFICIAL_FILE_SHA256 = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7",
    "model.safetensors.index.json": "ceaf1429bfa18eecc75d079737e8d1fe8dc0627f5ec7e6a985514626d1ac6a77",
    "tokenizer.json": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
    "tokenizer_config.json": "2f58f4bbd7bbce15d683f525954ef3a92cd82f5e06415a9c513859bf8ab72436",
    "model-00001-of-00004.safetensors": "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451",
    "model-00002-of-00004.safetensors": "dff635b0f6dbbaad2a2d633ef037ec0a39bc165cc1806c712fbd6fcbcb4526c0",
    "model-00003-of-00004.safetensors": "30b1d4c53d84eb018f642cad7b373f0aabf79699872d8702c1f38577c0a59a2f",
    "model-00004-of-00004.safetensors": "36cbc9c60375693629f25743c1e77ebb1724af58e671b2376463193c7fd21ef6",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_z(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    spread = centered.std(axis=1, keepdims=True, ddof=0)
    assert np.isfinite(spread).all() and (spread > 0).all()
    return centered / spread


def balanced(true: np.ndarray, predicted: np.ndarray, labels: list[str]) -> float:
    return float(np.mean([(predicted[true == label] == label).mean() for label in labels]))


def aligned_scores(path: Path, reference: pd.DataFrame, columns: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
    assert len(frame) == 2160 and frame.id.is_unique and set(frame.id) == set(reference.id)
    assert all(column in frame for column in columns)
    ordered = frame.set_index("id").loc[reference.id].reset_index()
    assert ordered[KEYS].equals(reference), path
    matrix = ordered[columns].to_numpy(dtype=np.float64)
    assert matrix.shape == (2160, 18) and np.isfinite(matrix).all()
    return matrix


def compare_csv(path: Path, rows: list[dict]) -> None:
    saved = pd.read_csv(path).to_dict(orient="records")
    assert len(saved) == len(rows)
    for actual, expected in zip(saved, rows):
        assert set(actual) == set(expected), path
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                assert abs(float(actual[key]) - float(value)) < 1e-12, (path, key)
            else:
                assert actual[key] == value, (path, key)


def main() -> None:
    cfg_path = ROOT / "configs/e30_qwen3_embedding8b.json"
    cfg = read_json(cfg_path)
    lock_path, resource_path = SNAP / "protocol_lock.json", SNAP / "resource_report.json"
    lock, resource = read_json(lock_path), read_json(resource_path)
    saved_metrics = read_json(SNAP / "analysis/metrics.json")
    assert lock["protocol_version"] == 3 and lock["stage"] == "protocol_lock_before_validation"
    assert lock["config_sha256"] == digest(cfg_path) == saved_metrics["config_sha256"]
    assert lock["script_sha256"] == digest(ROOT / "run_e30_supervised_embedding8b.py")
    assert saved_metrics["protocol_lock_sha256"] == digest(lock_path)
    assert resource["protocol_lock_sha256"] == digest(lock_path)
    assert lock["model_repo"] == cfg["model_repo"] == "Qwen/Qwen3-Embedding-8B"
    assert lock["model_revision"] == cfg["model_revision"] == resource["model_revision"]
    assert lock["model_license"] == cfg["model_license"] == "Apache-2.0"
    assert lock["train_sha256"] == cfg["official_train_sha256"]
    assert lock["e0_manifest_sha256"] == digest(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv")
    e0_path = ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv"
    assert lock["e0_oof_sha256"] == digest(e0_path)
    assert lock["development_rows"] == 2160 and lock["development_prompts"] == 120
    assert lock["fold_rows"] == {str(fold): 432 for fold in range(1, 6)}
    assert not lock["validation_accuracy_computed"]
    assert resource["model_file_sha256"] == OFFICIAL_FILE_SHA256
    assert resource["config_sha256"] == digest(cfg_path)
    assert resource["stage"] == "one_fold_resource_only"
    assert resource["training_rows"] == 1728 and resource["training_epochs"] == 1
    assert resource["unlabeled_inference_rows"] == 4000
    assert resource["deployment_model_count"] == 1
    assert resource["completed_optimizer_updates"] + resource["skipped_optimizer_updates"] == resource["planned_optimizer_updates"]
    resource_update_fraction = resource["completed_optimizer_updates"] / resource["planned_optimizer_updates"]
    assert resource_update_fraction >= cfg["resource_gate"]["min_completed_optimizer_update_fraction"]
    gib = 2**30
    assert resource["train_peak_reserved_bytes"] < cfg["resource_gate"]["max_train_peak_gib"] * gib
    assert resource["inference_peak_reserved_bytes"] < cfg["resource_gate"]["max_train_peak_gib"] * gib
    assert resource["train_seconds"] <= cfg["resource_gate"]["max_one_epoch_seconds"]
    assert resource["projected_three_epoch_seconds"] <= cfg["resource_gate"]["max_projected_three_epoch_seconds"]
    assert resource["inference_seconds"] <= cfg["resource_gate"]["max_four_thousand_inference_seconds"]
    assert all(resource["gate"].values()) and resource["gate_passed"]
    assert not resource["validation_accuracy_computed"] and not resource["hidden_test_runtime_proven"]

    e0 = pd.read_csv(e0_path, dtype=str, keep_default_na=False)
    assert len(e0) == 2160 and e0.id.is_unique and e0.prompt_id.nunique() == 120
    assert e0.true_model.nunique() == 18 and e0.true_model.value_counts().eq(120).all()
    assert e0.prompt_id.value_counts().eq(18).all()
    assert e0.groupby("prompt_id").fold.nunique().eq(1).all()
    assert e0.fold.value_counts().eq(432).all()
    manifest = pd.read_csv(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv", dtype=str)
    development_prompts = set(manifest.loc[manifest.split == "development", "prompt_id"])
    assert set(e0.prompt_id) == development_prompts and len(development_prompts) == 120
    assert manifest.split.eq("holdout").sum() == 30
    reference = e0[KEYS].copy()

    pieces, run_resources = [], []
    expected_packages = resource["packages"]
    for package in ("transformers", "peft", "accelerate", "bitsandbytes"):
        assert expected_packages[package] == cfg["package_versions"][package]
    assert expected_packages["torch"].startswith(cfg["package_versions"]["torch_prefix"])
    for fold in range(1, 6):
        folder = SNAP / f"fold_{fold}"
        path, run = folder / "oof_predictions.csv", read_json(folder / "run.json")
        assert run["experiment"] == cfg["experiment"] and run["fold"] == fold
        assert run["config_sha256"] == digest(cfg_path)
        assert run["protocol_lock_sha256"] == digest(lock_path)
        assert run["resource_report_sha256"] == digest(resource_path)
        assert run["model_revision"] == cfg["model_revision"]
        assert run["model_file_sha256"] == OFFICIAL_FILE_SHA256
        assert run["packages"] == expected_packages
        assert run["oof_predictions_sha256"] == digest(path)
        assert run["train_rows"] == 1728 and run["validation_rows"] == 432
        assert run["epochs"] == cfg["epochs"] == 3
        assert len(run["epoch_loss"]) == 3 and np.isfinite(run["epoch_loss"]).all()
        assert run["completed_optimizer_updates"] + run["skipped_optimizer_updates"] == run["planned_optimizer_updates"] == 648
        assert run["completed_optimizer_updates"] / run["planned_optimizer_updates"] >= 0.95
        assert run["no_epoch_or_threshold_selection"] and not run["holdout_evaluated"]
        piece = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
        assert len(piece) == 432 and piece.id.is_unique and set(piece.fold) == {str(fold)}
        expected = e0[e0.fold == str(fold)]
        assert set(piece.id) == set(expected.id)
        ordered = piece.set_index("id").loc[expected.id].reset_index()
        assert ordered[KEYS].equals(expected[KEYS].reset_index(drop=True))
        assert piece.prompt_id.nunique() == 24
        assert set(piece.prompt_id).isdisjoint(set(e0.loc[e0.fold != str(fold), "prompt_id"]))
        pieces.append(piece)
        run_resources.append({
            "fold": fold,
            "train_seconds": run["train_seconds"],
            "train_peak_reserved_bytes": run["train_peak_reserved_bytes"],
            "validation_seconds": run["validation_inference_seconds"],
            "validation_peak_reserved_bytes": run["validation_inference_peak_reserved_bytes"],
            "completed_updates": run["completed_optimizer_updates"],
            "planned_updates": run["planned_optimizer_updates"],
        })
    joined = pd.concat(pieces, ignore_index=True)
    assert joined.id.is_unique and set(joined.id) == set(e0.id)
    joined = joined.set_index("id").loc[e0.id].reset_index()
    saved_oof = pd.read_csv(SNAP / "analysis/oof_predictions.csv", dtype={key: str for key in KEYS}, keep_default_na=False)
    assert saved_oof.equals(joined)
    score_cols = [column for column in saved_oof if column.startswith("score_")]
    labels = [column.removeprefix("score_") for column in score_cols]
    assert len(labels) == 18 and labels == saved_metrics["labels"]
    assert set(labels) == set(e0.true_model)
    logits = saved_oof[score_cols].to_numpy(dtype=np.float64)
    assert logits.shape == (2160, 18) and np.isfinite(logits).all()
    label_array = np.asarray(labels)
    true = e0.true_model.to_numpy()
    single = label_array[logits.argmax(axis=1)]
    assert single.tolist() == saved_oof.pred_model.tolist()

    e24_metrics_path = ROOT / "outputs/e24_prvsiyan_analysis/metrics.json"
    assert digest(e24_metrics_path) == saved_metrics["e24_reference_metrics_sha256"]
    e24_metrics = read_json(e24_metrics_path)
    source_paths = {
        "E24 E0c": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
        "RoBERTa": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
        "ModernBERT seed 42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
        "ModernBERT seed 17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
        "ModernBERT seed 73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
    }
    matrices = {}
    for name, path in source_paths.items():
        assert digest(path) == saved_metrics["e24_source_score_sha256"][name]
        matrices[name] = aligned_scores(path, reference, score_cols)
    e24_oof_path = ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv"
    assert digest(e24_oof_path) == saved_metrics["e24_reference_oof_sha256"]
    e24_oof = pd.read_csv(e24_oof_path, dtype=str, keep_default_na=False)
    assert e24_oof[KEYS].equals(reference)
    base_logits = (row_z(matrices["E24 E0c"]) + row_z(matrices["RoBERTa"])
                   + np.mean(np.stack([row_z(matrices[f"ModernBERT seed {seed}"])
                                       for seed in (42, 17, 73)]), axis=0))
    old = label_array[base_logits.argmax(axis=1)]
    assert old.tolist() == e24_oof.pred_e16_with_e24.tolist()
    new = label_array[(base_logits + row_z(logits)).argmax(axis=1)]
    scores = {"e24_fixed": balanced(true, old, labels),
              "e30_single": balanced(true, single, labels),
              "e24_plus_e30_fixed": balanced(true, new, labels)}
    assert abs(scores["e24_fixed"] - 0.425) < 1e-12
    for name, score in scores.items():
        assert abs(score - saved_metrics["balanced_accuracy"][name]) < 1e-12

    folds = []
    for fold in range(1, 6):
        mask = e0.fold.to_numpy() == str(fold)
        old_ba, single_ba, new_ba = [balanced(true[mask], pred[mask], labels)
                                     for pred in (old, single, new)]
        folds.append({"fold": fold, "e24_fixed": old_ba, "e30_single": single_ba,
                      "e24_plus_e30_fixed": new_ba, "fusion_delta": new_ba - old_ba})
    classes = []
    for label in labels:
        mask = true == label
        old_recall, single_recall, new_recall = [float((pred[mask] == label).mean())
                                                 for pred in (old, single, new)]
        classes.append({"model": label, "support": int(mask.sum()),
                        "e24_fixed_recall": old_recall,
                        "e30_single_recall": single_recall,
                        "e24_plus_e30_fixed_recall": new_recall,
                        "fusion_delta": new_recall - old_recall})
    compare_csv(SNAP / "analysis/fold_metrics.csv", folds)
    compare_csv(SNAP / "analysis/class_metrics.csv", classes)
    index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((18, 18), dtype=int)
    for actual, predicted in zip(true, single):
        confusion[index[actual], index[predicted]] += 1
    assert confusion.tolist() == saved_metrics["e30_single_confusion_matrix"]
    gains = int(((old != true) & (new == true)).sum())
    losses = int(((old == true) & (new != true)).sum())
    assert gains == saved_metrics["fusion_wrong_to_right"]
    assert losses == saved_metrics["fusion_right_to_wrong"]

    # Exactly one row per class in each prompt makes paired BA the mean prompt
    # hit delta. This implementation is independent of the training summarizer.
    prompt_deltas = pd.DataFrame({"prompt": e0.prompt_id,
                                  "delta": (new == true).astype(int) - (old == true).astype(int)})
    by_prompt = prompt_deltas.groupby("prompt", sort=True).delta.sum().to_numpy()
    choices = np.random.default_rng(cfg["seed"]).integers(0, 120, size=(20_000, 120))
    draws = by_prompt[choices].sum(axis=1) / 2160
    ci = np.quantile(draws, [0.025, 0.975]).tolist()
    assert np.allclose(ci, saved_metrics["paired_prompt_bootstrap_e24_plus_e30_minus_e24"]["confidence_interval_95"], atol=1e-12)

    delta = scores["e24_plus_e30_fixed"] - scores["e24_fixed"]
    gate_cfg = cfg["promotion_gate"]
    gate = {
        "delta_at_least_0_01": delta >= gate_cfg["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": sum(row["fusion_delta"] >= -1e-12 for row in folds)
        >= gate_cfg["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": min(row["fusion_delta"] for row in classes)
        > -gate_cfg["max_allowed_class_recall_drop"] + 1e-12,
    }
    assert gate == saved_metrics["gate"]
    assert all(gate.values()) == saved_metrics["gate_passed_against_e24"]
    assert not saved_metrics["holdout_evaluated"] and not saved_metrics["fusion_weight_tuned"]
    e28_audit = read_json(ROOT / "docs/e28_qwen8b_fivefold_audit_aggregate.json")
    assert abs(e28_audit["scores"]["e24_fixed"] - scores["e24_fixed"]) < 1e-12
    e28_folds = {row["fold"]: row for row in e28_audit["folds"]}
    e28_classes = {row["model"]: row for row in e28_audit["classes_recall"]}
    relative_e28_fold_deltas = [row["e24_plus_e30_fixed"]
                                - e28_folds[row["fold"]]["e24_plus_e28_fixed"] for row in folds]
    relative_e28_class_deltas = {row["model"]: row["e24_plus_e30_fixed_recall"]
                                 - e28_classes[row["model"]]["e24_plus_e28_fixed_recall"] for row in classes}
    relative_e28_delta = scores["e24_plus_e30_fixed"] - e28_audit["scores"]["e24_plus_e28_fixed"]
    relative_e28_gate = {
        "delta_at_least_0_01": relative_e28_delta >= gate_cfg["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": sum(value >= -1e-12 for value in relative_e28_fold_deltas)
        >= gate_cfg["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": min(relative_e28_class_deltas.values())
        > -gate_cfg["max_allowed_class_recall_drop"] + 1e-12,
    }
    result = {
        "scope": "independent audit of E30 official development OOF; no holdout/test response read",
        "protocol_lock_sha256": digest(lock_path),
        "summary_oof_sha256": digest(SNAP / "analysis/oof_predictions.csv"),
        "summary_metrics_sha256": digest(SNAP / "analysis/metrics.json"),
        "e24_reference_oof_sha256": digest(e24_oof_path),
        "protocol_code_config_hashes_match": True,
        "e24_source_hashes_match": True,
        "fold_oof_hashes_and_metadata_match": True,
        "fold_validation_prompts_disjoint_and_match_e0": True,
        "rows": 2160, "prompts": 120, "classes": 18,
        "resource_gate_passed": True,
        "resource_update_fraction": resource_update_fraction,
        "resource_train_seconds": resource["train_seconds"],
        "resource_inference_4000_seconds": resource["inference_seconds"],
        "resource_train_peak_reserved_bytes": resource["train_peak_reserved_bytes"],
        "resource_inference_peak_reserved_bytes": resource["inference_peak_reserved_bytes"],
        "fold_resources": run_resources,
        "total_fold_train_seconds": sum(row["train_seconds"] for row in run_resources),
        "max_fold_train_peak_reserved_bytes": max(row["train_peak_reserved_bytes"] for row in run_resources),
        "max_fold_validation_peak_reserved_bytes": max(row["validation_peak_reserved_bytes"] for row in run_resources),
        "scores": scores, "folds": folds, "class_recalls": classes,
        "fusion_delta": delta, "wrong_to_right": gains, "right_to_wrong": losses,
        "paired_prompt_bootstrap": {"repeats": 20_000, "seed": cfg["seed"], "ci_95": ci},
        "gate": gate, "gate_passed_against_e24": all(gate.values()),
        "exploratory_comparison_to_unpromoted_e28_fixed_fusion": {
            "balanced_accuracy_delta": relative_e28_delta,
            "fold_deltas": relative_e28_fold_deltas,
            "worst_class": min(relative_e28_class_deltas, key=relative_e28_class_deltas.get),
            "worst_class_recall_delta": min(relative_e28_class_deltas.values()),
            "same_gate_if_e28_is_treated_as_strongest_fixed_comparator": relative_e28_gate,
        },
        "conservative_gate_passed_if_all_prior_fixed_fusions_count": all(relative_e28_gate.values()),
        "hidden_test_runtime_proven": False,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scores", "fusion_delta", "folds", "gate")}, indent=2))


if __name__ == "__main__":
    main()
