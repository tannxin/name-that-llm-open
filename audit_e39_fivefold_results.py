#!/usr/bin/env python3
"""Independently audit the complete E39 Recovery1 official fivefold OOF.

The snapshot is Git ignored. This script reads development OOF scores only after
checking that all five fold outputs and the final summary exist. It never reads
holdout answers, model weights, or Kaggle test data. Only aggregate findings are
written to the public-safe output JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = ROOT / "outputs/e39_recovery1_audit_snapshot"
DEFAULT_OUTPUT = ROOT / "docs/e39_qwen3_base_fivefold_audit_aggregate.json"
KEYS = ["id", "prompt_id", "true_model", "fold"]
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
    "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3",
    "glm-5.3-flash", "kimi-k3",
]
SCORES = [f"score_{name}" for name in LABELS]
EXP = "e39_qwen3_8b_base_official_only_fivefold"
BOOT = "6bff0b13-b608-4f8e-9072-97427e329d61"
GPU = "GPU-528ca5ce-dadc-c24a-12ef-e1ece5fb5d1c"
PIN = {
    "train": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
    "e0_manifest": "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15",
    "e0_oof": "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa",
    "e24_metrics": "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557",
    "e24_oof": "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448",
    "prereg": "234d214dff5caaa4688e9e4d12c96083b316b7985a4a73050ad2a94f47ac78fd",
    "config": "044dff60abc6a3f7022883c788d2c84b72886b8557d1e7f056e5d6429d0c203b",
    "script": "fdb343b776e54506c167eb5181ee299ffa43b1f29a59647f9bb0cfe8c45d57a1",
    "launcher": "1a4412e7e44dc7a6a4cd311451aaae11c7e051163c38513c360f3538d91de0fe",
    "worker": "717430e919d2cc156b47a7e501b0441abaed9783c595b4ccc9bb3d5b78e65408",
    "monitor": "c994cf69a1642919c1e39c1879e13a45be9d85733fb19c4a838c1a2b3fe8e68d",
    "outer": "8f9053f1b275a46625c44c68770a3de9b4e5b0fd44bbf2a2d3dcfcdaa4199d34",
    "resource_lock": "c59c4f8115b4da7a5df81282e66ffd1d079723c819f71d0c950e40aee234937a",
    "resource_report": "0fe04ea70d07e53d1d354157eb7da9935947e9c641ca50791dbbc10fc2121d1f",
    "old_abort": "476a6f43e149e4360544a96c63cebc37eec1336f1d5e1283a205791439369231",
    "e28_audit": "da593d5f335fb137fc88a1b35aea670edaa63cce9496e34e22caeca8f6c2e3dc",
    "e30_a_audit": "4094c41f15703b1a94d64ebcec245832b3783e9622b738ceb4199c0c5df345ec",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash(path: Path, expected: str, name: str) -> None:
    require(digest(path) == expected, f"Frozen {name} SHA-256 differs: {path}")


def close(actual: float, expected: float, name: str, atol: float = 1e-12) -> None:
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol,
            f"{name}: {actual!r} != {expected!r}")


def balanced(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[true == label] == label).mean() for label in LABELS]))


def zscore(matrix: np.ndarray) -> np.ndarray:
    require(matrix.shape == (2160, 18) and np.isfinite(matrix).all(), "Invalid score matrix")
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True, ddof=0)
    require(np.isfinite(scale).all() and np.all(scale > 0), "Flat/non-finite score row")
    return centered / scale


def read_scores(path: Path, reference: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
    require(len(frame) == 2160 and frame.id.is_unique and set(frame.id) == set(reference.id),
            f"Score identity coverage differs: {path}")
    require(all(key in frame for key in [*KEYS, *SCORES]), f"Missing score column: {path}")
    frame = frame.set_index("id").loc[reference.id].reset_index()
    require(frame[KEYS].equals(reference), f"Score metadata differs: {path}")
    matrix = frame[SCORES].to_numpy(dtype=np.float64)
    require(matrix.shape == (2160, 18) and np.isfinite(matrix).all(),
            f"Invalid score values: {path}")
    return frame, matrix


def compare_rows(path: Path, expected: list[dict]) -> None:
    rows = pd.read_csv(path).to_dict(orient="records")
    require(len(rows) == len(expected), f"Metric CSV row count differs: {path}")
    for actual, wanted in zip(rows, expected):
        require(set(actual) == set(wanted), f"Metric CSV columns differ: {path}")
        for key, value in wanted.items():
            if isinstance(value, (int, float)):
                close(actual[key], value, f"{path.name} {key}")
            else:
                require(actual[key] == value, f"{path.name} {key} differs")


def promotion(old_score: float, old_folds: list[float], old_class: list[float],
              new_score: float, new_folds: list[float], new_class: list[float],
              gate: dict, reference: str) -> dict:
    fold_delta = [fresh - prior for prior, fresh in zip(old_folds, new_folds)]
    class_delta = [fresh - prior for prior, fresh in zip(old_class, new_class)]
    checks = {
        "delta_at_least_0_01": new_score - old_score >= gate["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": sum(x >= -1e-12 for x in fold_delta)
        >= gate["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": min(class_delta)
        > -gate["max_allowed_class_recall_drop"] + 1e-12,
    }
    return {
        "reference": reference,
        "balanced_accuracy_delta": new_score - old_score,
        "fold_deltas": fold_delta,
        "folds_non_decreasing": sum(x >= -1e-12 for x in fold_delta),
        "class_recall_deltas": dict(zip(LABELS, class_delta)),
        "worst_class": LABELS[int(np.argmin(class_delta))],
        "worst_class_recall_delta": min(class_delta),
        "gate": checks,
        "gate_passed": all(checks.values()),
    }


def main(snapshot: Path, output: Path) -> None:
    # Fail closed on partial artifacts before opening any E39 validation score.
    required = [snapshot / name for name in (
        "analysis/metrics.json", "analysis/fold_metrics.csv", "analysis/class_metrics.csv",
        "analysis/oof_predictions.csv", "fivefold/protocol_lock.json",
        "fivefold/gpu2_release_status.json", "fivefold/gpu2_released.ok",
        "queue/status.json", "queue/launch_intent.json", "queue/launcher.live.log",
        "resource_preflight/protocol_lock.json", "resource_preflight/resource_report.json",
        "model/e39_import_report.json", "model/e39_mirror_manifest.json")]
    for fold in range(1, 6):
        required.extend(snapshot / f"fold_{fold}" / name for name in (
            "run.json", "oof_predictions.csv", "import_manifest.json"))
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Incomplete fivefold snapshot: " + ", ".join(missing))
    cfg = read_json(ROOT / "configs/e39_qwen3_8b_base_fivefold.json")
    gate = cfg["promotion_gate"]
    require(cfg["experiment"] == EXP and cfg["protocol_version"] == 1
            and cfg["model_repo"] == "Qwen/Qwen3-8B-Base"
            and cfg["model_revision"] == "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
            and cfg["folds"] == [1, 2, 3, 4, 5]
            and cfg["seed"] == 42 and cfg["epochs"] == 3
            and cfg["gradient_accumulation"] == 8 and cfg["micro_batch_size"] == 1
            and cfg["evaluation"] == "final_epoch_only_no_epoch_or_threshold_selection"
            and cfg["fixed_fusion"] ==
            "row_z_E24_E0c_plus_row_z_RoBERTa_plus_mean_three_row_z_ModernBERT_plus_1x_row_z_E39_logits",
            "Frozen protocol differs")
    require(gate == {"reference": "fixed_E24_fusion", "fixed_added_branch_weight": 1.0,
                     "min_balanced_accuracy_delta": 0.01, "min_non_decreasing_folds": 4,
                     "max_allowed_class_recall_drop": 0.05,
                     "paired_prompt_bootstrap_repeats": 20000}, "Promotion gate changed")
    for path, name in (
        (ROOT / "data/train.csv", "train"),
        (ROOT / "outputs/e0_tfidf_svm/split_manifest.csv", "e0_manifest"),
        (ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv", "e0_oof"),
        (ROOT / "outputs/e24_prvsiyan_analysis/metrics.json", "e24_metrics"),
        (ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv", "e24_oof"),
        (ROOT / "docs/e39_qwen3_8b_base_preregistration.md", "prereg"),
        (ROOT / "configs/e39_qwen3_8b_base_fivefold.json", "config"),
        (ROOT / "run_e39_qwen3_base_fivefold.py", "script"),
        (ROOT / "launch_e39_qwen3_base_fivefold_recovery1.py", "launcher"),
        (ROOT / "outputs/e39_base_download/fivefold_worker_recovery1.sh", "worker"),
        (ROOT / "outputs/e39_base_download/fivefold_monitor_recovery1.sh", "monitor"),
        (ROOT / "outputs/e39_base_download/fivefold_launch_recovery1.sh", "outer"),
        (snapshot / "resource_preflight/protocol_lock.json", "resource_lock"),
        (snapshot / "resource_preflight/resource_report.json", "resource_report"),
        (ROOT / "docs/e28_qwen8b_fivefold_audit_aggregate.json", "e28_audit"),
        (ROOT / "docs/e30_embedding8b_fivefold_audit_aggregate.json", "e30_a_audit"),
    ):
        check_hash(path, PIN[name], name)

    lock_path = snapshot / "fivefold/protocol_lock.json"
    lock = read_json(lock_path)
    resource_lock = read_json(snapshot / "resource_preflight/protocol_lock.json")
    resource = read_json(snapshot / "resource_preflight/resource_report.json")
    mirror_path = snapshot / "model/e39_mirror_manifest.json"
    mirror = read_json(mirror_path)
    model_path = snapshot / "model/e39_import_report.json"
    model = read_json(model_path)
    require(lock["experiment"] == EXP and lock["protocol_version"] == 1
            and lock["folds"] == [1, 2, 3, 4, 5]
            and lock["model_revision"] == cfg["model_revision"]
            and lock["config_sha256"] == PIN["config"]
            and lock["script_sha256"] == PIN["script"]
            and lock["preregistration_sha256"] == PIN["prereg"]
            and lock["train_sha256"] == PIN["train"]
            and lock["e0_manifest_sha256"] == PIN["e0_manifest"]
            and lock["e0_oof_sha256"] == PIN["e0_oof"]
            and lock["e24_metrics_sha256"] == PIN["e24_metrics"]
            and lock["resource_report_sha256"] == PIN["resource_report"]
            and lock["resource_protocol_lock_sha256"] == PIN["resource_lock"]
            and lock["model_import_report_sha256"] == digest(model_path)
            and lock["resource_script_sha256"] == digest(ROOT / "run_e39_qwen3_base_resource.py"),
            "Fivefold protocol lock differs")
    require(resource_lock["script_sha256"] == lock["resource_script_sha256"]
            and resource_lock["preregistration_sha256"] == PIN["prereg"]
            and resource_lock["train_sha256"] == PIN["train"]
            and resource_lock["e0_split_manifest_sha256"] == PIN["e0_manifest"]
            and resource_lock["model_import_report_sha256"] == digest(model_path)
            and resource_lock["model_repo"] == cfg["model_repo"]
            and resource_lock["model_revision"] == cfg["model_revision"]
            and resource_lock["validation_accuracy_allowed"] is False,
            "Resource protocol lock differs")
    require(model["all_sha256_verified"] and mirror["all_sha256_verified"]
            and model["scratch_manifest_sha256"] == digest(mirror_path)
            and model["hf_revision"] == mirror["hf_revision"] == cfg["model_revision"]
            and model["model_id"] == mirror["model_id"] == cfg["model_repo"],
            "Pinned model import differs")
    files = mirror["files"]
    hashes = {name: entry["sha256"] for name, entry in files.items()}
    sizes = {name: entry["size"] for name, entry in files.items()}
    require(len(files) == 12 and hashes == resource_lock["pinned_model_file_sha256"]
            and all(size > 0 for size in sizes.values()), "Model file manifest differs")
    require(resource["gate_passed"] and len(resource["gate"]) == 7
            and all(resource["gate"].values())
            and resource["validation_accuracy_computed"] is False
            and resource["validation_scores_saved"] is False
            and resource["kaggle_hardware_equivalence_proven"] is False
            and resource["training_rows"] == 1728 and resource["inference_rows"] == 4000
            and resource["training_epochs_completed"] == 1
            and resource["physical_gpu_uuid"] == GPU
            and resource["model_files_bytes"] == sizes
            and resource["resource_protocol_lock_sha256"] == PIN["resource_lock"]
            and resource["model_import_report_sha256"] == digest(model_path),
            "Resource probe differs")
    require(resource["completed_optimizer_updates"] + resource["skipped_optimizer_updates"]
            == resource["planned_one_epoch_optimizer_updates"]
            and resource["completed_optimizer_updates"]
            >= math.ceil(.95 * resource["planned_one_epoch_optimizer_updates"])
            and 0 < resource["train_peak_reserved_bytes"] < 22 * 2**30
            and 0 < resource["inference_peak_reserved_bytes"] < 22 * 2**30
            and 0 < resource["training_seconds"] <= 3600
            and 0 < resource["projected_three_epoch_seconds"] <= 10800
            and 0 < resource["inference_seconds"] <= 10800,
            "Resource measurements exceed preregistered limits")
    require(resource["packages"] == {
        "torch": resource["packages"]["torch"], "transformers": "4.51.3",
        "peft": "0.15.2", "accelerate": "1.6.0", "bitsandbytes": "0.45.5",
    } and resource["packages"]["torch"].startswith("2.6.0"),
            "Resource runtime package versions differ")

    intent = read_json(snapshot / "queue/launch_intent.json")
    queue = read_json(snapshot / "queue/status.json")
    release = read_json(snapshot / "fivefold/gpu2_release_status.json")
    launcher_log = (snapshot / "queue/launcher.live.log").read_text(encoding="utf-8")
    require(intent["experiment"] == queue["experiment"] == release["experiment"] == EXP
            and intent["pod_boot_id"] == queue["pod_boot_id"] == BOOT
            and intent["worker_sha256"] == PIN["worker"]
            and intent["monitor_sha256"] == PIN["monitor"]
            and intent["original_abort_status_sha256"] == PIN["old_abort"]
            and queue["exit_code"] == 0 and queue["stage"] == "guarded_launcher"
            and release["e39_success"] is True
            and release["gpu2_idle_at_release"] is True and release["gpu2_uuid"] == GPU
            and (snapshot / "fivefold/gpu2_released.ok").read_text().strip()
            == "E39_GPU2_RELEASED"
            and "E39_GPU2_IDLE_TEN_MINUTES_CONFIRMED" in launcher_log
            and "E39_FIVEFOLD_COMPLETE" in launcher_log
            and "E39_FIVEFOLD_EXIT 0" in launcher_log,
            "Guarded recovery did not finish cleanly")
    require(launcher_log.count("IMPORT_COMPLETE") == 5,
            "Not all five folds imported in guarded launcher")
    abort = ROOT / "outputs/e39_base_download/abort1_20261111T1745Z"
    prior_attestation = read_json(abort / "readonly_attestation.json")
    require(prior_attestation["abort_before_preflight_or_fit"] is True
            and prior_attestation["scratch_files"] == []
            and set(prior_attestation["output_files"]) == {
                "gpu2_release_status.json", "gpu2_released.ok", "runner.lock"}
            and prior_attestation["status"]["exit_code"] == 1
            and prior_attestation["release"]["e39_success"] is False,
            "First preflight-only abort evidence differs")
    for filename, source in (
        ("status.json", "outputs/e39_qwen3_base_queue/status.json"),
        ("launcher.live.log", "outputs/e39_qwen3_base_queue/launcher.live.log"),
        ("gpu2_release_status.json", "outputs/e39_qwen3_base_fivefold/gpu2_release_status.json"),
    ):
        check_hash(abort / filename, prior_attestation["evidence_sha256"][source],
                   f"original abort {filename}")
    require(digest(abort / "status.json") == PIN["old_abort"],
            "Recovery launch refers to a different original abort")

    e0 = pd.read_csv(ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv",
                     dtype=str, keep_default_na=False)
    manifest = pd.read_csv(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv",
                           dtype=str, keep_default_na=False)
    require(len(e0) == 2160 and e0.id.is_unique and e0.prompt_id.nunique() == 120
            and e0.true_model.nunique() == 18 and set(e0.true_model) == set(LABELS)
            and e0.prompt_id.value_counts().eq(18).all()
            and e0.true_model.value_counts().eq(120).all()
            and e0.fold.value_counts().eq(432).all()
            and e0.groupby("prompt_id").fold.nunique().eq(1).all(),
            "E0 development identity/fold invalid")
    development = set(manifest.loc[manifest.split == "development", "prompt_id"])
    holdout = set(manifest.loc[manifest.split == "holdout", "prompt_id"])
    require(len(development) == 120 and len(holdout) == 30
            and not (development & holdout) and set(e0.prompt_id) == development,
            "Holdout boundary differs")
    reference = e0[KEYS].copy()
    lock_hash = digest(lock_path)
    fold_frames = []
    fold_resources = []
    for fold in range(1, 6):
        folder = snapshot / f"fold_{fold}"
        oof_path = folder / "oof_predictions.csv"
        run_path = folder / "run.json"
        run = read_json(run_path)
        imported = read_json(folder / "import_manifest.json")
        require(imported["experiment"] == EXP and imported["fold"] == fold
                and imported["files"]["oof_predictions.csv"] == digest(oof_path)
                and imported["files"]["run.json"] == digest(run_path)
                and "adapter/adapter_model.safetensors" in imported["files"]
                and all(len(value) == 64 for value in imported["files"].values()),
                f"Fold {fold} hash-checked import differs")
        require(run["experiment"] == EXP and run["fold"] == fold and run["seed"] == 42
                and run["model_revision"] == cfg["model_revision"]
                and run["model_required_file_sha256"] == hashes
                and run["model_required_file_sizes"] == sizes
                and run["protocol_lock_sha256"] == lock_hash
                and run["resource_report_sha256"] == PIN["resource_report"]
                and run["oof_predictions_sha256"] == digest(oof_path)
                and run["physical_gpu_uuid"] == GPU
                and run["packages"] == resource["packages"]
                and run["train_rows"] == 1728 and run["validation_rows"] == 432
                and run["training_epochs_completed"] == 3
                and run["no_epoch_or_threshold_selection"] is True
                and run["holdout_evaluated"] is False,
                f"Fold {fold} protocol differs")
        planned = 3 * math.ceil(1728 / 8)
        require(run["planned_optimizer_updates"] == planned
                and run["completed_optimizer_updates"] + run["skipped_optimizer_updates"]
                == planned and run["completed_optimizer_updates"] >= math.ceil(.95 * planned)
                and len(run["epoch_loss"]) == 3
                and all(math.isfinite(value) for value in run["epoch_loss"])
                and 0 < run["train_peak_reserved_bytes"] < 22 * 2**30
                and 0 < run["validation_peak_reserved_bytes"] < 22 * 2**30
                and 0 < run["train_seconds"] <= 10800
                and 0 < run["validation_inference_seconds"] <= 10800,
                f"Fold {fold} update/resource gate differs")
        piece = pd.read_csv(oof_path, dtype={key: str for key in KEYS},
                            keep_default_na=False)
        expected = e0.loc[e0.fold == str(fold), KEYS].reset_index(drop=True)
        require(len(piece) == 432 and piece.id.is_unique
                and set(piece.id) == set(expected.id)
                and set(piece.fold) == {str(fold)}
                and piece.prompt_id.nunique() == 24
                and not (set(piece.prompt_id) & holdout),
                f"Fold {fold} validation identities differ")
        piece = piece.set_index("id").loc[expected.id].reset_index()
        require(piece[KEYS].equals(expected)
                and (piece.seed == 42).all()
                and piece.groupby("prompt_id").true_model.nunique().eq(18).all(),
                f"Fold {fold} row metadata differs")
        logits = piece[SCORES].to_numpy(dtype=np.float64)
        require(logits.shape == (432, 18) and np.isfinite(logits).all()
                and np.array_equal(np.asarray(LABELS)[logits.argmax(axis=1)],
                                   piece.pred_model.to_numpy()),
                f"Fold {fold} score/prediction invalid")
        fold_frames.append(piece)
        fold_resources.append({key: run[key] for key in (
            "fold", "planned_optimizer_updates", "completed_optimizer_updates",
            "skipped_optimizer_updates", "train_seconds", "train_peak_reserved_bytes",
            "validation_inference_seconds", "validation_peak_reserved_bytes")})

    combined = pd.concat(fold_frames, ignore_index=True)
    require(len(combined) == 2160 and combined.id.is_unique
            and set(combined.id) == set(reference.id), "Incomplete or duplicate fivefold OOF")
    combined = combined.set_index("id").loc[reference.id].reset_index()
    summary_frame, logits = read_scores(snapshot / "analysis/oof_predictions.csv", reference)
    require(summary_frame[KEYS + ["seed", "pred_model"]].equals(
        combined[KEYS + ["seed", "pred_model"]])
        and np.array_equal(logits, combined[SCORES].to_numpy(dtype=np.float64)),
        "Aggregate OOF differs from five imported folds")
    true = reference.true_model.to_numpy()
    names = np.asarray(LABELS)
    single = names[logits.argmax(axis=1)]
    require(np.array_equal(single, summary_frame.pred_model.to_numpy()),
            "Aggregate predictions differ from logits")

    e24_metrics = read_json(ROOT / "outputs/e24_prvsiyan_analysis/metrics.json")
    sources = {
        "e24_scores": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
        "roberta_scores": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
        "modern_seed_42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
        "modern_seed_17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
        "modern_seed_73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
    }
    expected_source_hashes = {
        "e24_scores": e24_metrics["e24_scores_sha256"],
        "roberta_scores": e24_metrics["roberta_scores_sha256"],
        **{f"modern_seed_{seed}": e24_metrics["modern_seed_score_sha256"][str(seed)]
           for seed in (42, 17, 73)},
    }
    require(lock["e24_sources_sha256"] == {"e24_oof": PIN["e24_oof"],
                                            **expected_source_hashes},
            "Frozen E24 source hashes differ")
    matrices = {}
    for name, path in sources.items():
        check_hash(path, expected_source_hashes[name], name)
        _, matrices[name] = read_scores(path, reference)
    baseline_scores = (zscore(matrices["e24_scores"]) + zscore(matrices["roberta_scores"])
                       + np.mean(np.stack([zscore(matrices[f"modern_seed_{seed}"])
                                           for seed in (42, 17, 73)]), axis=0))
    base = names[baseline_scores.argmax(axis=1)]
    archived_e24 = pd.read_csv(ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv",
                               dtype=str, keep_default_na=False)
    require(archived_e24[KEYS].equals(reference)
            and np.array_equal(base, archived_e24.pred_e16_with_e24.to_numpy()),
            "Fixed E24 baseline does not reproduce archive")
    close(balanced(true, base), .425, "E24 fixed balanced accuracy")
    added = names[(baseline_scores + zscore(logits)).argmax(axis=1)]
    scores = {"e24_fixed": balanced(true, base), "e39_single": balanced(true, single),
              "e24_plus_e39_fixed": balanced(true, added)}
    fold_rows = []
    for fold in range(1, 6):
        mask = reference.fold.to_numpy() == str(fold)
        old, new = balanced(true[mask], base[mask]), balanced(true[mask], added[mask])
        fold_rows.append({"fold": fold, "e24_fixed": old,
                          "e39_single": balanced(true[mask], single[mask]),
                          "e24_plus_e39_fixed": new, "fusion_delta": new - old})
    class_rows = []
    for label in LABELS:
        mask = true == label
        old, new = float((base[mask] == label).mean()), float((added[mask] == label).mean())
        class_rows.append({"model": label, "support": int(mask.sum()),
                           "e24_fixed_recall": old,
                           "e39_single_recall": float((single[mask] == label).mean()),
                           "e24_plus_e39_fixed_recall": new, "fusion_delta": new - old})
    compare_rows(snapshot / "analysis/fold_metrics.csv", fold_rows)
    compare_rows(snapshot / "analysis/class_metrics.csv", class_rows)
    saved = read_json(snapshot / "analysis/metrics.json")
    require(saved["experiment"] == EXP and saved["stage"] == "five_fold_oof_analysis"
            and saved["protocol_lock_sha256"] == lock_hash
            and saved["resource_report_sha256"] == PIN["resource_report"]
            and saved["e24_reference_oof_sha256"] == PIN["e24_oof"]
            and saved["rows"] == 2160 and saved["prompts"] == 120
            and saved["labels"] == LABELS and saved["holdout_evaluated"] is False
            and saved["fusion_weight_tuned"] is False
            and saved["comparison_reference_must_be_rechecked_at_promotion"] is True,
            "Saved aggregate protocol differs")
    for name, value in scores.items():
        close(saved["balanced_accuracy"][name], value, name)
    confusion = np.zeros((18, 18), dtype=np.int64)
    index = {label: i for i, label in enumerate(LABELS)}
    for actual, predicted in zip(true, single):
        confusion[index[actual], index[predicted]] += 1
    require(saved["e39_single_confusion_matrix"] == confusion.tolist(),
            "Saved confusion matrix differs")
    require(saved["fusion_wrong_to_right"] == int(((base != true) & (added == true)).sum())
            and saved["fusion_right_to_wrong"] == int(((base == true) & (added != true)).sum()),
            "Saved error transitions differ")
    delta = scores["e24_plus_e39_fixed"] - scores["e24_fixed"]
    close(saved["fusion_delta"], delta, "Saved E39 fusion delta")
    old_folds = [row["e24_fixed"] for row in fold_rows]
    new_folds = [row["e24_plus_e39_fixed"] for row in fold_rows]
    old_class = [row["e24_fixed_recall"] for row in class_rows]
    new_class = [row["e24_plus_e39_fixed_recall"] for row in class_rows]
    against_e24 = promotion(scores["e24_fixed"], old_folds, old_class,
                            scores["e24_plus_e39_fixed"], new_folds, new_class,
                            gate, "E24 fixed")
    require(saved["gate"] == against_e24["gate"]
            and saved["gate_passed_against_e24"] == against_e24["gate_passed"]
            and saved["folds_non_decreasing"] == against_e24["folds_non_decreasing"],
            "Saved preregistered promotion gate differs")
    close(saved["worst_class_recall_delta"],
          against_e24["worst_class_recall_delta"], "Saved worst class change")
    prompt_deltas = pd.DataFrame({"prompt_id": reference.prompt_id,
                                  "delta": (added == true).astype(int)
                                  - (base == true).astype(int)}).groupby(
                                      "prompt_id", sort=True).delta.sum().to_numpy()
    require(len(prompt_deltas) == 120 and int(prompt_deltas.sum())
            == int((added == true).sum() - (base == true).sum()),
            "Paired prompt bootstrap identity differs")
    rng = np.random.default_rng(42)
    draws = np.empty(20_000, dtype=np.float64)
    for start in range(0, len(draws), 500):
        stop = min(start + 500, len(draws))
        sample = rng.integers(0, 120, size=(stop - start, 120))
        draws[start:stop] = prompt_deltas[sample].sum(axis=1) / 2160
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    bootstrap = saved["paired_prompt_bootstrap_e24_plus_e39_minus_e24"]
    require(bootstrap["bootstrap_unit"] == "prompt_id"
            and bootstrap["repeats"] == 20_000 and bootstrap["seed"] == 42,
            "Saved bootstrap protocol differs")
    close(bootstrap["difference"], delta, "Saved bootstrap point estimate")
    for actual, wanted in zip(bootstrap["confidence_interval_95"], interval):
        close(actual, wanted, "Saved bootstrap interval")

    def prior_audit(name: str, path: Path, score_key: str,
                    fold_key: str, class_key: str, collection: str) -> dict:
        prior = read_json(path)
        require(prior["rows"] == 2160 and prior["prompts"] == 120
                and prior["classes"] == 18
                and prior["e24_reference_oof_sha256"] == PIN["e24_oof"],
                f"{name} independent audit reference differs")
        close(prior["scores"]["e24_fixed"], scores["e24_fixed"],
              f"{name} E24 baseline")
        fold_map = {int(row["fold"]): row[fold_key] for row in prior["folds"]}
        class_map = {row["model"]: row[class_key] for row in prior[collection]}
        require(set(fold_map) == set(range(1, 6)) and set(class_map) == set(LABELS),
                f"{name} fold/class coverage differs")
        return promotion(prior["scores"][score_key],
                         [fold_map[fold] for fold in range(1, 6)],
                         [class_map[label] for label in LABELS],
                         scores["e24_plus_e39_fixed"], new_folds, new_class,
                         gate, name)

    against_e28 = prior_audit(
        "E24+E28 fixed, unpromoted",
        ROOT / "docs/e28_qwen8b_fivefold_audit_aggregate.json",
        "e24_plus_e28_fixed", "e24_plus_e28_fixed",
        "e24_plus_e28_fixed_recall", "classes_recall")
    against_e30a = prior_audit(
        "E24+E30-A fixed, unpromoted",
        ROOT / "docs/e30_embedding8b_fivefold_audit_aggregate.json",
        "e24_plus_e30_fixed", "e24_plus_e30_fixed",
        "e24_plus_e30_fixed_recall", "class_recalls")
    conservative = (against_e24["gate_passed"] and against_e28["gate_passed"]
                    and against_e30a["gate_passed"])
    result = {
        "scope": "Independent complete E39 official development audit; holdout/test excluded",
        "infrastructure_recovery": "Recovery1 after preflight-only GPU2 race; no original fold outputs",
        "original_abort_attestation_sha256": digest(abort / "readonly_attestation.json"),
        "rows": 2160, "prompts": 120, "classes": 18, "holdout_prompts_untouched": 30,
        "protocol_lock_sha256": lock_hash,
        "resource_report_sha256": PIN["resource_report"],
        "summary_oof_sha256": digest(snapshot / "analysis/oof_predictions.csv"),
        "summary_metrics_sha256": digest(snapshot / "analysis/metrics.json"),
        "e24_reference_oof_sha256": PIN["e24_oof"],
        "model_import_report_sha256": digest(model_path),
        "model_manifest_sha256": digest(mirror_path),
        "fivefold_protocol_data_model_code_hashes_match": True,
        "fivefold_oof_hashes_fold_prompt_class_integrity_match": True,
        "resource_gate_passed": True,
        "fold_resources": fold_resources,
        "total_fold_train_seconds": sum(row["train_seconds"] for row in fold_resources),
        "max_fold_train_peak_reserved_bytes": max(
            row["train_peak_reserved_bytes"] for row in fold_resources),
        "max_fold_validation_peak_reserved_bytes": max(
            row["validation_peak_reserved_bytes"] for row in fold_resources),
        "scores": scores, "folds": fold_rows, "class_recalls": class_rows,
        "e39_single_confusion_matrix": confusion.tolist(),
        "wrong_to_right": saved["fusion_wrong_to_right"],
        "right_to_wrong": saved["fusion_right_to_wrong"],
        "paired_prompt_bootstrap": {"unit": "prompt_id", "seed": 42,
                                    "repeats": 20_000,
                                    "confidence_interval_95": interval},
        "comparison_e24": against_e24,
        "comparison_e28": against_e28,
        "comparison_e30_a": against_e30a,
        "preregistered_gate_passed_against_e24": against_e24["gate_passed"],
        "conservative_gate_passed_all_prior_fixed_fusions": conservative,
        "kaggle_t4x2_9h_deployment_proven": False,
    }
    require(not output.exists(), f"Refusing to overwrite existing audit: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "scores", "comparison_e24", "comparison_e28", "comparison_e30_a",
        "paired_prompt_bootstrap", "preregistered_gate_passed_against_e24",
        "conservative_gate_passed_all_prior_fixed_fusions")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.snapshot, args.output)
