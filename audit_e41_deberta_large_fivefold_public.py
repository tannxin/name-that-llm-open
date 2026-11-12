"""Reproduce the locked E41-v2 official fivefold audit and fixed E24 fusion.

Only archived small CSV/JSON metadata are read. The official training CSV is
hashed, never parsed here. No holdout examples, model weights, or GPU are used.
Supply independently reviewed host identifiers via CLI, never from the snapshot
being audited. Optional private frozen files can be verified without publishing
the original machine-specific training configuration or launcher.
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
SNAP = ROOT / "outputs/e41_fivefold_audit_snapshot_v2"
RESOURCE = ROOT / "outputs/e41_resource_preflight"
OUT = ROOT / "outputs/e41_fivefold_public_audit_result.json"
KEYS = ["id", "prompt_id", "true_model", "fold"]
EXPERIMENT = "e41_deberta_v3_large_official_only_fivefold"
LOCK_SHA = "a87aad9da02e2b308ec457ae2f45909e219f26261dab32524a97ec2c84bfa49e"
CONFIG_SHA = "a7547d292e6f093b17b68706c9f275078160028f3b2be08b6c539db28e36d861"
RUNNER_SHA = "a73eb962a6e6b070375f4fbeea3ce3ea0070ccd343fb91602f07eb6a8ecdbadc"
LAUNCHER_SHA = "b917186e49c33778640a3408ade4b5b5c8a0b17abe11a8fd94cc8c16e8a9f8f6"
CONTROL_SHA = "796a2cf2763556dedf6f3d75284a614445f341ec591b775ddeb3be4901a7a6f9"
RESOURCE_SHA = "e2431ae27c5d4016f98c081472898e94c0304ce2989a02156fc386c2e908b633"
RESOURCE_PREFLIGHT_SHA = "866c1a8e3f08414e772b7155c36a8af4698640881ff51d521411c0e8e684330b"
MODEL_MANIFEST_SHA = "d683b8dc579d1ac26b52f508751901627c087dcba3634d23fdf96c0849ad2b73"
E24_METRICS_SHA = "301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557"
E24_OOF_SHA = "8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448"
TRAINING = {
    "seed": 42, "epochs": 12, "batch_size": 16, "max_length": 256,
    "pooling": "mean", "dropout": .1, "encoder_learning_rate": .00002,
    "head_learning_rate": .001, "warmup_ratio": .1, "weight_decay": .01,
    "gradient_clip_norm": 1.0,
    "loss": "inverse_fold_train_frequency_weighted_18_class_ce",
    "precision": "fp16_amp_gradscaler",
    "scheduler": "linear_after_successful_optimizer_step",
    "epoch_selection": "fixed_final_epoch", "save_model_checkpoint": False,
}
FUSION_AUDIT = {
    "baseline": "E24",
    "candidate": "row_z(E24_E0c)+row_z(RoBERTa)+mean_seed_row_z(ModernBERT)+1*row_z(E41)",
    "minimum_balanced_accuracy_gain": .01,
    "minimum_non_decreasing_folds": 4,
    "maximum_allowed_class_recall_drop_exclusive": .05,
    "paired_prompt_bootstrap_replicates": 20_000,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, wanted: float, name: str, tolerance: float = 1e-10) -> None:
    require(math.isfinite(actual) and abs(actual - wanted) <= tolerance,
            f"{name}: {actual} != {wanted}")


def read_oof(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def score_matrix(path: Path, reference: pd.DataFrame, columns: list[str]) -> np.ndarray:
    frame = read_oof(path)
    require(len(frame) == 2160 and frame.id.is_unique and set(frame.id) == set(reference.id),
            f"Incomplete score identities: {path}")
    require(all(column in frame for column in columns), f"Missing score column: {path}")
    frame = frame.set_index("id").loc[reference.id].reset_index()
    require(frame[KEYS].equals(reference), f"Score row metadata differs: {path}")
    matrix = frame[columns].to_numpy(dtype=np.float64)
    require(matrix.shape == (2160, 18) and np.isfinite(matrix).all(),
            f"Nonfinite or incomplete score matrix: {path}")
    return matrix


def row_z(matrix: np.ndarray) -> np.ndarray:
    scale = matrix.std(axis=1, keepdims=True)
    require(np.isfinite(scale).all() and (scale > 0).all(), "Invalid row standard deviation")
    return (matrix - matrix.mean(axis=1, keepdims=True)) / scale


def balanced(true: np.ndarray, predicted: np.ndarray, labels: list[str]) -> float:
    return float(np.mean([(predicted[true == label] == label).mean() for label in labels]))


def main(expected_gpu_uuid: str, expected_boot_id: str,
         verify_private_frozen_files: bool) -> None:
    cfg_path = ROOT / "configs/e41_deberta_v3_large_fivefold_v2.json"
    lock_path = SNAP / "protocol_lock.json"
    lock = read_json(lock_path)
    require(digest(lock_path) == LOCK_SHA, "E41 CPU lock SHA differs")
    mirror = read_json(RESOURCE / "e41_mirror_manifest.json")
    cfg = {
        "protocol_version": "e41_fivefold_v2", "experiment": EXPERIMENT,
        "official_only": True, "folds": [1, 2, 3, 4, 5],
        "model_id": "microsoft/deberta-v3-large",
        "model_revision": "64a8c8eab3e352a784c658aef62be1662607476f",
        "model_files": mirror["files"],
        "source_files": lock["source_file_sha256"],
        "official_files": lock["official_file_sha256"],
        "physical_gpu_uuid": expected_gpu_uuid,
        "training": TRAINING, "fusion_audit": FUSION_AUDIT,
    }
    source_hashes = {
        "run_finetune_encoder.py": ROOT / "run_finetune_encoder.py",
        "run_e0.py": ROOT / "run_e0.py",
        "e41_deberta_v3_large_preregistration.md":
            ROOT / "docs/e41_deberta_v3_large_preregistration.md",
    }
    for path, wanted in (
        (RESOURCE / "resource_result_v2.json", RESOURCE_SHA),
        (RESOURCE / "preflight_v2.json", RESOURCE_PREFLIGHT_SHA),
        (RESOURCE / "e41_mirror_manifest.json", MODEL_MANIFEST_SHA),
        (ROOT / "outputs/e24_prvsiyan_analysis/metrics.json", E24_METRICS_SHA),
        (ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv", E24_OOF_SHA),
    ):
        require(digest(path) == wanted, f"Frozen hash differs: {path}")
    if verify_private_frozen_files:
        for path, wanted in (
            (cfg_path, CONFIG_SHA),
            (ROOT / "run_e41_deberta_large_fivefold_v2.py", RUNNER_SHA),
            (ROOT / "launch_e41_deberta_large_fivefold_v2.py", LAUNCHER_SHA),
            (ROOT / "launch_e41_deberta_large_control_v2.py", CONTROL_SHA),
        ):
            require(digest(path) == wanted, f"Private frozen file hash differs: {path}")
        private_cfg = read_json(cfg_path)
        for name, value in cfg.items():
            require(private_cfg[name] == value, f"Private frozen config differs: {name}")
    require(cfg["protocol_version"] == lock["protocol_version"] == "e41_fivefold_v2",
            "Protocol version changed")
    require(cfg["experiment"] == lock["experiment"] == EXPERIMENT, "Experiment changed")
    require(cfg["official_only"] and cfg["folds"] == [1, 2, 3, 4, 5], "Official folds changed")
    require(cfg["model_revision"] == lock["model_revision"], "Model revision changed")
    require(cfg["model_id"] == lock["model_id"] == "microsoft/deberta-v3-large",
            "Model ID changed")
    require(lock["config_sha256"] == CONFIG_SHA and lock["fold_runner_sha256"] == RUNNER_SHA
            and lock["launcher_sha256"] == LAUNCHER_SHA, "Code lock changed")
    require(lock["resource_report_sha256"] == RESOURCE_SHA
            and lock["resource_preflight_sha256"] == RESOURCE_PREFLIGHT_SHA
            and lock["model_manifest_sha256"] == MODEL_MANIFEST_SHA,
            "Resource/model lock changed")
    for name, path in source_hashes.items():
        require(digest(path) == cfg["source_files"][name]
                == lock["source_file_sha256"][name], f"Source hash differs: {name}")
    for name, wanted in cfg["official_files"].items():
        require(digest(ROOT / name) == wanted == lock["official_file_sha256"][name],
                f"Official file hash differs: {name}")
    require(mirror["all_sha256_verified"] and mirror["hf_revision"] == cfg["model_revision"]
            and mirror["files"] == cfg["model_files"], "Pinned model manifest differs")
    resource = read_json(RESOURCE / "resource_result_v2.json")
    require(resource["status"] == "pass" and all(resource["gates"].values())
            and resource["validation_score_computed"] is False, "Resource probe invalid")
    require(resource["gpu"]["uuid"] == cfg["physical_gpu_uuid"] == expected_gpu_uuid,
            "GPU0 UUID changed")
    v1 = read_json(RESOURCE / "resource_result_attempt1.json")
    require(v1["status"] == "failed" and v1["exception_type"] == "RuntimeError"
            and "non-finite" in v1["exception"] and v1["validation_score_computed"] is False,
            "Original resource failure evidence differs")

    intent = read_json(SNAP / "launch_intent.json")
    finished = read_json(SNAP / "launch_finished.json")
    complete = read_json(SNAP / "fivefold_complete.json")
    require(intent["approved_lock_sha256"] == finished["protocol_lock_sha256"]
            == complete["protocol_lock_sha256"] == LOCK_SHA, "Launch lock differs")
    require(intent["pod_boot_id"] == expected_boot_id
            and intent["gpu0_uuid"] == expected_gpu_uuid,
            "Launch host differs")
    require(intent["config_sha256"] == CONFIG_SHA and intent["runner_sha256"] == RUNNER_SHA
            and intent["launcher_sha256"] == LAUNCHER_SHA
            and intent["control_sha256"] == CONTROL_SHA, "Launch code differs")
    require(finished["launch_intent_sha256"] == digest(SNAP / "launch_intent.json")
            and finished["state"] == "complete" and finished["launcher_exit_code"] == 0
            and finished["error"] is None and complete["all_five_folds_imported"]
            and complete["gpu0_released"], "Fivefold launch did not finish cleanly")

    e0_path = ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv"
    e0 = read_oof(e0_path)
    manifest = pd.read_csv(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv",
                           dtype=str, keep_default_na=False)
    require(len(e0) == 2160 and e0.id.is_unique and e0.prompt_id.nunique() == 120
            and e0.true_model.nunique() == 18, "Official development identity invalid")
    require(e0.true_model.value_counts().eq(120).all()
            and e0.prompt_id.value_counts().eq(18).all(), "Unbalanced dev rows")
    require(e0.groupby("prompt_id").fold.nunique().eq(1).all()
            and e0.fold.value_counts().eq(432).all(), "Official folds invalid")
    development = set(manifest.loc[manifest.split == "development", "prompt_id"])
    holdout = set(manifest.loc[manifest.split == "holdout", "prompt_id"])
    require(len(development) == 120 and len(holdout) == 30
            and not development & holdout and set(e0.prompt_id) == development,
            "Holdout split contaminated")
    require(lock["development_rows"] == 2160 and lock["development_prompts"] == 120
            and lock["holdout_prompts_untouched"] == 30, "CPU lock counts invalid")
    require(lock["training"] == cfg["training"] and lock["validation_epoch_selection"]
            == "fixed_final_only", "Training protocol changed")
    reference = e0[KEYS].copy()

    summary = read_json(SNAP / "analysis/summary.json")
    analysis_manifest = read_json(SNAP / "analysis/import_manifest.json")
    expected_files = {name: digest(SNAP / "analysis" / name) for name in
                      ("oof_predictions.csv", "fold_metrics.csv", "class_metrics.csv",
                       "summary.json")}
    require(analysis_manifest["experiment"] == EXPERIMENT
            and analysis_manifest["protocol_lock_sha256"] == LOCK_SHA
            and analysis_manifest["files"] == expected_files, "Analysis import invalid")
    require(complete["summary_sha256"] == expected_files["summary.json"],
            "Completion summary hash invalid")
    require(summary["experiment"] == EXPERIMENT and summary["protocol_version"]
            == "e41_fivefold_v2" and summary["config_sha256"] == CONFIG_SHA
            and summary["protocol_lock_sha256"] == LOCK_SHA
            and summary["fold_runner_sha256"] == RUNNER_SHA
            and summary["files"] == {key: expected_files[key] for key in
                                      ("oof_predictions.csv", "fold_metrics.csv",
                                       "class_metrics.csv")}
            and summary["n_oof_rows"] == 2160 and summary["n_oof_prompts"] == 120
            and summary["holdout_touched"] is False
            and summary["fixed_final_epoch_only"] is True
            and summary["fusion_gate_evaluated"] is False, "Fivefold summary invalid")

    pieces: list[pd.DataFrame] = []
    resources: list[dict] = []
    for fold in range(1, 6):
        folder = SNAP / f"fold_{fold}"
        run = read_json(folder / "run.json")
        import_manifest = read_json(folder / "import_manifest.json")
        path = folder / "oof_predictions.csv"
        require(import_manifest["experiment"] == EXPERIMENT
                and import_manifest["fold"] == fold
                and import_manifest["protocol_lock_sha256"] == LOCK_SHA
                and import_manifest["files"] == {
                    "oof_predictions.csv": digest(path), "run.json": digest(folder / "run.json")},
                f"Fold {fold} import invalid")
        require(run["experiment"] == EXPERIMENT and run["protocol_version"]
                == "e41_fivefold_v2" and run["fold"] == fold
                and run["config_sha256"] == CONFIG_SHA
                and run["protocol_lock_sha256"] == LOCK_SHA
                and run["fold_runner_sha256"] == RUNNER_SHA
                and run["resource_report_sha256"] == RESOURCE_SHA
                and run["model_revision"] == cfg["model_revision"]
                and run["oof_predictions_sha256"] == digest(path),
                f"Fold {fold} protocol/hash invalid")
        require(run["gpu"]["physical_index"] == 0
                and run["gpu"]["uuid"] == expected_gpu_uuid
                and run["gpu"]["memory_used_mib_before"] <= 100
                and run["gpu"]["utilization_percent_before"] <= 5,
                f"Fold {fold} GPU ownership invalid")
        require(run["training_epochs_completed"] == cfg["training"]["epochs"] == 12
                and run["train_rows"] == 1728 and run["train_prompts"] == 96
                and run["validation_rows"] == 432 and run["validation_prompts"] == 24
                and run["planned_optimizer_steps"] == 1296
                and run["completed_optimizer_steps"] + run["skipped_optimizer_steps"]
                == 1296 and run["completed_optimizer_steps"] / 1296 >= .95
                and len(run["epoch_training_loss"]) == 12
                and np.isfinite(run["epoch_training_loss"]).all()
                and run["validation_inference_after_fixed_final_epoch_only"] is True
                and run["model_checkpoint_saved"] is False,
                f"Fold {fold} training protocol/resources invalid")
        require(0 < run["train_peak_reserved_bytes"] < 22 * (1 << 30)
                and 0 < run["validation_peak_reserved_bytes"] < 22 * (1 << 30)
                and 0 < run["train_seconds"] / 12 < 900
                and 0 < run["validation_inference_seconds"] < 7200,
                f"Fold {fold} resource gate invalid")
        piece = read_oof(path)
        require(len(piece) == 432 and piece.id.is_unique
                and set(piece.fold) == {str(fold)}
                and piece.prompt_id.nunique() == 24
                and not set(piece.prompt_id) & holdout,
                f"Fold {fold} validation rows invalid")
        expected = e0.loc[e0.fold == str(fold), KEYS]
        require(set(piece.id) == set(expected.id)
                and piece.set_index("id").loc[expected.id].reset_index()[KEYS].equals(
                    expected.reset_index(drop=True)), f"Fold {fold} identity mismatch")
        pieces.append(piece)
        resources.append({key: run[key] for key in (
            "fold", "training_epochs_completed", "planned_optimizer_steps",
            "completed_optimizer_steps", "skipped_optimizer_steps", "train_seconds",
            "train_peak_reserved_bytes", "validation_inference_seconds",
            "validation_peak_reserved_bytes")})
    combined = pd.concat(pieces, ignore_index=True)
    require(combined.id.is_unique and set(combined.id) == set(e0.id),
            "Fivefold OOF has missing/duplicate IDs")
    combined = combined.set_index("id").loc[e0.id].reset_index()
    saved = read_oof(SNAP / "analysis/oof_predictions.csv")
    require(saved.drop(columns="pred_model").equals(combined)
            and saved[KEYS].equals(reference),
            "Aggregate OOF differs from imported folds/E0")
    columns = [column for column in saved if column.startswith("score_")]
    labels = [column[6:] for column in columns]
    require(len(labels) == 18 and len(set(labels)) == 18
            and set(labels) == set(e0.true_model), "E41 class columns invalid")
    logits = saved[columns].to_numpy(dtype=np.float64)
    require(logits.shape == (2160, 18) and np.isfinite(logits).all(),
            "E41 OOF logits invalid")
    label_array = np.asarray(labels)
    true = e0.true_model.to_numpy()
    single = label_array[logits.argmax(axis=1)]
    require(np.array_equal(single, saved.pred_model.to_numpy()),
            "E41 predicted classes differ from logits")
    single_ba = balanced(true, single, labels)
    close(single_ba, summary["overall_balanced_accuracy"], "E41 single BA")
    fold_metrics = pd.read_csv(SNAP / "analysis/fold_metrics.csv").to_dict(orient="records")
    class_metrics = pd.read_csv(SNAP / "analysis/class_metrics.csv").to_dict(orient="records")
    require(len(fold_metrics) == len(summary["fold_metrics"]) == 5
            and len(class_metrics) == len(summary["class_metrics"]) == 18,
            "Summary metric lengths invalid")
    for fold in range(1, 6):
        mask = reference.fold.to_numpy() == str(fold)
        score = balanced(true[mask], single[mask], labels)
        for row in (fold_metrics[fold - 1], summary["fold_metrics"][fold - 1]):
            require(int(row["fold"]) == fold, f"Fold {fold} metric order invalid")
            close(score, row["balanced_accuracy"], f"Fold {fold} E41 BA")
            for key in resources[fold - 1]:
                if key != "fold" and key != "training_epochs_completed" and key != "planned_optimizer_steps":
                    saved_key = key
                    if key in row:
                        close(float(resources[fold - 1][key]), float(row[saved_key]),
                              f"Fold {fold} {saved_key}")
    for row, summary_row, label in zip(class_metrics, summary["class_metrics"], labels):
        mask = true == label
        correct = int((single[mask] == label).sum())
        require(row["model"] == summary_row["model"] == label
                and int(row["support"]) == summary_row["support"] == 120
                and int(row["correct"]) == summary_row["correct"] == correct,
                f"Class {label} metrics invalid")
        close(correct / 120, row["recall"], f"Class {label} saved recall")
        close(correct / 120, summary_row["recall"], f"Class {label} summary recall")

    e24_metrics = read_json(ROOT / "outputs/e24_prvsiyan_analysis/metrics.json")
    e24_sources = {
        "E24 E0c": (ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
                    e24_metrics["e24_scores_sha256"]),
        "RoBERTa": (ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
                    e24_metrics["roberta_scores_sha256"]),
        **{f"ModernBERT seed {seed}": (
            ROOT / f"outputs/e{15 if seed == 42 else 16}_modernbert_"
                   f"{('base_12ep_eager' if seed == 42 else 'seed' + str(seed) + '_eager')}/oof_predictions.csv",
            e24_metrics["modern_seed_score_sha256"][str(seed)]) for seed in (42, 17, 73)},
    }
    matrices = {}
    for name, (path, wanted) in e24_sources.items():
        require(digest(path) == wanted, f"E24 archived score hash differs: {name}")
        matrices[name] = score_matrix(path, reference, columns)
    e24_saved = read_oof(ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv")
    require(e24_saved[KEYS].equals(reference), "E24 archived identity mismatch")
    baseline = (row_z(matrices["E24 E0c"]) + row_z(matrices["RoBERTa"])
                + np.mean(np.stack([row_z(matrices[f"ModernBERT seed {seed}"])
                                    for seed in (42, 17, 73)]), axis=0))
    old = label_array[baseline.argmax(axis=1)]
    require(np.array_equal(old, e24_saved.pred_e16_with_e24.to_numpy()),
            "Fixed E24 blend does not reproduce archive")
    old_ba = balanced(true, old, labels)
    close(old_ba, .425, "E24 BA")
    candidate = label_array[(baseline + row_z(logits)).argmax(axis=1)]
    candidate_ba = balanced(true, candidate, labels)
    fold_rows = []
    for fold in range(1, 6):
        mask = reference.fold.to_numpy() == str(fold)
        e24_fold = balanced(true[mask], old[mask], labels)
        e41_fold = balanced(true[mask], single[mask], labels)
        fusion_fold = balanced(true[mask], candidate[mask], labels)
        fold_rows.append({"fold": fold, "e24_fixed": e24_fold, "e41_single": e41_fold,
                          "e24_plus_e41_fixed": fusion_fold,
                          "fusion_delta": fusion_fold - e24_fold})
    class_rows = []
    for label in labels:
        mask = true == label
        prior = float((old[mask] == label).mean())
        fresh = float((single[mask] == label).mean())
        fused = float((candidate[mask] == label).mean())
        class_rows.append({"model": label, "support": int(mask.sum()),
                           "e24_fixed_recall": prior, "e41_single_recall": fresh,
                           "e24_plus_e41_fixed_recall": fused,
                           "fusion_delta": fused - prior})
    delta = candidate_ba - old_ba
    gate = cfg["fusion_audit"]
    require(gate["baseline"] == "E24" and gate["candidate"]
            == "row_z(E24_E0c)+row_z(RoBERTa)+mean_seed_row_z(ModernBERT)+1*row_z(E41)"
            and gate["paired_prompt_bootstrap_replicates"] == 20_000,
            "Pre-registered fusion rule changed")
    gate_checks = {
        "delta_at_least_0_01": delta >= gate["minimum_balanced_accuracy_gain"] - 1e-12,
        "at_least_four_folds_non_decreasing": sum(
            row["fusion_delta"] >= -1e-12 for row in fold_rows)
            >= gate["minimum_non_decreasing_folds"],
        "every_class_recall_drop_strictly_below_0_05": min(
            row["fusion_delta"] for row in class_rows)
            > -gate["maximum_allowed_class_recall_drop_exclusive"] + 1e-12,
    }
    per_prompt = pd.DataFrame({"prompt_id": reference.prompt_id,
                               "delta": (candidate == true).astype(int)
                               - (old == true).astype(int)})
    per_prompt_delta = per_prompt.groupby("prompt_id", sort=True).delta.sum().to_numpy()
    require(len(per_prompt_delta) == 120 and sum(per_prompt_delta) ==
            int((candidate == true).sum() - (old == true).sum()),
            "Prompt paired differences invalid")
    bootstrap_indices = np.random.default_rng(42).integers(0, 120, size=(20_000, 120))
    bootstrap_draws = per_prompt_delta[bootstrap_indices].sum(axis=1) / 2160
    ci = np.quantile(bootstrap_draws, [.025, .975])
    wrong_to_right = int(((old != true) & (candidate == true)).sum())
    right_to_wrong = int(((old == true) & (candidate != true)).sum())

    e14_path = ROOT / "outputs/e14_deberta_v3_base_12ep_256_http/oof_predictions.csv"
    e14_logits = score_matrix(e14_path, reference, columns)
    e14_single = label_array[e14_logits.argmax(axis=1)]
    result = {
        "scope": "independent E41-v2 official development audit; holdout/test excluded",
        "protocol_lock_sha256": LOCK_SHA,
        "summary_sha256": expected_files["summary.json"],
        "oof_sha256": expected_files["oof_predictions.csv"],
        "e24_reference_oof_sha256": E24_OOF_SHA,
        "model_manifest_sha256": MODEL_MANIFEST_SHA,
        "locked_model_data_hashes_match": True,
        "private_frozen_code_config_verified": verify_private_frozen_files,
        "locked_config_runner_launcher_control_sha256": {
            "config": CONFIG_SHA, "runner": RUNNER_SHA,
            "launcher": LAUNCHER_SHA, "control": CONTROL_SHA},
        "all_five_fold_imports_and_resources_match": True,
        "holdout_prompts_untouched": 30,
        "rows": 2160, "prompts": 120, "classes": 18,
        "scores": {"e24_fixed": old_ba, "e14_base_single_diagnostic":
                   balanced(true, e14_single, labels), "e41_large_single": single_ba,
                   "e24_plus_e41_fixed": candidate_ba},
        "folds": fold_rows,
        "classes_recall": class_rows,
        "fold_resources": resources,
        "total_train_seconds": sum(row["train_seconds"] for row in resources),
        "max_fold_peak_reserved_bytes": max(row["train_peak_reserved_bytes"]
                                            for row in resources),
        "fusion_delta": delta,
        "wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong,
        "paired_prompt_bootstrap": {"repeats": 20_000, "seed": 42,
                                    "unit": "prompt_id", "confidence_interval_95": ci.tolist()},
        "gate": gate_checks, "gate_passed_against_e24": all(gate_checks.values()),
        "v1_resource_attempt": {"status": "failed_due_to_strict_nonfinite_clip_guard",
                                "report_sha256": digest(RESOURCE / "resource_result_attempt1.json"),
                                "validation_score_computed": False},
        "v2_resource_probe_passed": True,
        "kaggle_equivalence_or_public_score_inferred": False,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: result[name] for name in
                      ("rows", "prompts", "scores", "fusion_delta", "folds",
                       "paired_prompt_bootstrap", "gate", "gate_passed_against_e24")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu-uuid", required=True,
                        help="GPU UUID from an independently reviewed launch record")
    parser.add_argument("--expected-boot-id", required=True,
                        help="Host boot ID from an independently reviewed launch record")
    parser.add_argument("--verify-private-frozen-files", action="store_true",
                        help="Also hash-check unpublished original config/runner/launcher/control")
    args = parser.parse_args()
    main(args.expected_gpu_uuid, args.expected_boot_id,
         args.verify_private_frozen_files)
