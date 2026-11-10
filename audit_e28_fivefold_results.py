"""Independently audit archived E28 development OOF; write aggregates only.

First copy the small E28 CSV/JSON outputs into the Git-ignored local snapshot
directory. This script never opens the official training CSV or holdout rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "outputs/e28_qwen8b_fivefold_audit_snapshot"
OUT = ROOT / "docs/e28_qwen8b_fivefold_audit_aggregate.json"
KEYS = ["id", "prompt_id", "true_model", "fold"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def standardize(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    assert np.isfinite(scale).all() and (scale > 0).all()
    return centered / scale


def ba(true: np.ndarray, pred: np.ndarray, labels: list[str]) -> float:
    return float(np.mean([(pred[true == label] == label).mean() for label in labels]))


def aligned_scores(path: Path, reference: pd.DataFrame, score_cols: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={key: str for key in KEYS}, keep_default_na=False)
    assert len(frame) == 2160 and frame.id.is_unique and set(frame.id) == set(reference.id)
    assert all(col in frame for col in score_cols)
    frame = frame.set_index("id").loc[reference.id].reset_index()
    for key in KEYS:
        assert frame[key].tolist() == reference[key].tolist(), (path.name, key)
    scores = frame[score_cols].to_numpy(dtype=np.float64)
    assert scores.shape == (2160, 18) and np.isfinite(scores).all()
    return scores


def main() -> None:
    lock = read(SNAP / "protocol_lock.json")
    metrics = read(SNAP / "analysis/metrics.json")
    resource = read(SNAP / "resource_report.json")
    release = read(SNAP / "gpu2_release_status.json")
    cfg = read(ROOT / "configs/e28_qwen3_8b_fivefold.json")
    assert lock["config_sha256"] == digest(ROOT / "configs/e28_qwen3_8b_fivefold.json")
    assert lock["script_sha256"] == digest(ROOT / "run_e28_qwen8b_fivefold.py")
    assert lock["resource_script_sha256"] == digest(ROOT / "run_e28_qwen8b_lora_feasibility.py")
    assert metrics["protocol_lock_sha256"] == digest(SNAP / "protocol_lock.json")
    assert metrics["resource_report_sha256"] == digest(SNAP / "resource_report.json")
    assert resource["gate_passed"] and resource["model_sha256_verified"]
    assert not resource["kaggle_hardware_equivalence_proven"]
    assert release["e28_success"] and release["gpu2_idle_at_release"]

    e0_path = ROOT / "outputs/e0_tfidf_svm/oof_predictions.csv"
    assert digest(e0_path) == lock["e0_oof_sha256"]
    assert digest(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv") == lock["e0_manifest_sha256"]
    e0 = pd.read_csv(e0_path, dtype=str, keep_default_na=False)
    assert len(e0) == 2160 and e0.id.is_unique and e0.prompt_id.nunique() == 120
    assert e0.true_model.nunique() == 18 and e0.true_model.value_counts().eq(120).all()
    assert e0.prompt_id.value_counts().eq(18).all()
    assert e0.groupby("prompt_id").fold.nunique().eq(1).all()
    assert e0.fold.value_counts().eq(432).all()
    manifest = pd.read_csv(ROOT / "outputs/e0_tfidf_svm/split_manifest.csv", dtype=str)
    development = set(manifest.loc[manifest.split == "development", "prompt_id"])
    assert len(development) == 120 and set(e0.prompt_id) == development
    assert manifest.split.eq("holdout").sum() == 30
    reference = e0[KEYS].copy()

    pieces, run_rows, first_model_hashes = [], [], None
    for fold in range(1, 6):
        folder = SNAP / f"fold_{fold}"
        path, run = folder / "oof_predictions.csv", read(folder / "run.json")
        assert run["oof_predictions_sha256"] == digest(path)
        assert run["protocol_lock_sha256"] == metrics["protocol_lock_sha256"]
        assert run["resource_report_sha256"] == metrics["resource_report_sha256"]
        assert run["model_revision"] == lock["model_revision"]
        assert run["fold"] == fold and run["seed"] == cfg["seed"]
        assert run["train_rows"] == 1728 and run["validation_rows"] == 432
        assert run["training_epochs_completed"] == cfg["epochs"] == 3
        assert run["no_epoch_or_threshold_selection"] and not run["holdout_evaluated"]
        assert run["physical_gpu_uuid"] == resource["physical_gpu_uuid"]
        assert run["packages"] == resource["packages"]
        for package in ("transformers", "peft", "accelerate", "bitsandbytes"):
            assert run["packages"][package] == cfg["package_versions"][package]
        assert run["packages"]["torch"].startswith(cfg["package_versions"]["torch_prefix"])
        assert len(run["epoch_loss"]) == 3 and np.isfinite(run["epoch_loss"]).all()
        assert run["completed_optimizer_updates"] + run["skipped_optimizer_updates"] == run["planned_optimizer_updates"]
        assert run["completed_optimizer_updates"] / run["planned_optimizer_updates"] >= 0.95
        if first_model_hashes is None:
            first_model_hashes = run["model_required_file_sha256"]
        assert run["model_required_file_sha256"] == first_model_hashes
        piece = pd.read_csv(path, dtype={key: str for key in [*KEYS, "seed"]}, keep_default_na=False)
        assert len(piece) == 432 and piece.id.is_unique
        assert set(piece.fold) == {str(fold)} and set(piece.seed) == {str(cfg["seed"])}
        expected = e0[e0.fold == str(fold)]
        assert set(piece.id) == set(expected.id)
        assert set(piece.prompt_id) == set(expected.prompt_id)
        assert piece.prompt_id.nunique() == 24
        assert set(piece.prompt_id).isdisjoint(set(e0.loc[e0.fold != str(fold), "prompt_id"]))
        ordered = piece.set_index("id").loc[expected.id].reset_index()
        assert ordered["prompt_id"].tolist() == expected.prompt_id.tolist()
        assert ordered["true_model"].tolist() == expected.true_model.tolist()
        pieces.append(piece)
        run_rows.append({
            "fold": fold, "train_rows": run["train_rows"], "validation_rows": run["validation_rows"],
            "epochs": run["training_epochs_completed"],
            "updates_completed": run["completed_optimizer_updates"],
            "updates_planned": run["planned_optimizer_updates"],
            "train_seconds": run["train_seconds"],
            "train_peak_reserved_bytes": run["train_peak_reserved_bytes"],
            "validation_seconds": run["validation_inference_seconds"],
            "validation_peak_reserved_bytes": run["validation_peak_reserved_bytes"],
        })
    concat = pd.concat(pieces, ignore_index=True)
    assert concat.id.is_unique and set(concat.id) == set(e0.id)
    concat = concat.set_index("id").loc[e0.id].reset_index()
    saved = pd.read_csv(SNAP / "analysis/oof_predictions.csv", dtype={key: str for key in [*KEYS, "seed"]}, keep_default_na=False)
    assert saved.equals(concat)
    score_cols = [col for col in saved if col.startswith("score_")]
    labels = [col[6:] for col in score_cols]
    assert len(labels) == 18 and labels == metrics["labels"] and set(labels) == set(e0.true_model)
    e28_scores = saved[score_cols].to_numpy(dtype=np.float64)
    assert e28_scores.shape == (2160, 18) and np.isfinite(e28_scores).all()
    true = e0.true_model.to_numpy()
    single = np.asarray(labels)[e28_scores.argmax(axis=1)]
    assert single.tolist() == saved.pred_model.tolist()

    e24_metrics_path = ROOT / "outputs/e24_prvsiyan_analysis/metrics.json"
    assert digest(e24_metrics_path) == lock["e24_metrics_sha256"]
    sources = {
        "e24_oof": ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv",
        "e24_scores": ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv",
        "roberta_scores": ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv",
        "modern_seed_42": ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
        "modern_seed_17": ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
        "modern_seed_73": ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
    }
    for name, path in sources.items():
        assert digest(path) == lock["e24_sources_sha256"][name], name
    assert digest(sources["e24_oof"]) == metrics["e24_reference_oof_sha256"]
    source_scores = {name: aligned_scores(path, reference, score_cols)
                     for name, path in sources.items() if name != "e24_oof"}
    baseline_scores = (
        standardize(source_scores["e24_scores"])
        + standardize(source_scores["roberta_scores"])
        + np.mean(np.stack([standardize(source_scores[f"modern_seed_{seed}"])
                            for seed in (42, 17, 73)]), axis=0)
    )
    old = np.asarray(labels)[baseline_scores.argmax(axis=1)]
    e24_saved = pd.read_csv(sources["e24_oof"], dtype=str, keep_default_na=False)
    assert e24_saved[KEYS].equals(reference)
    assert old.tolist() == e24_saved.pred_e16_with_e24.tolist()
    new = np.asarray(labels)[(baseline_scores + standardize(e28_scores)).argmax(axis=1)]

    scores = {"e24_fixed": ba(true, old, labels), "e28_single": ba(true, single, labels),
              "e24_plus_e28_fixed": ba(true, new, labels)}
    for name, value in scores.items():
        assert abs(value - metrics["balanced_accuracy"][name]) < 1e-12
    assert abs(scores["e24_fixed"] - 0.425) < 1e-12
    fold_rows = []
    for fold in range(1, 6):
        mask = e0.fold.to_numpy() == str(fold)
        row = {"fold": fold, "e24_fixed": ba(true[mask], old[mask], labels),
               "e28_single": ba(true[mask], single[mask], labels),
               "e24_plus_e28_fixed": ba(true[mask], new[mask], labels)}
        difference = row["e24_plus_e28_fixed"] - row["e24_fixed"]
        row["fusion_delta"] = 0.0 if abs(difference) < 1e-12 else difference
        fold_rows.append(row)
    class_rows = []
    for label in labels:
        mask = true == label
        row = {"model": label, "support": int(mask.sum()),
               "e24_fixed_recall": float((old[mask] == label).mean()),
               "e28_single_recall": float((single[mask] == label).mean()),
               "e24_plus_e28_fixed_recall": float((new[mask] == label).mean())}
        row["fusion_delta"] = row["e24_plus_e28_fixed_recall"] - row["e24_fixed_recall"]
        class_rows.append(row)
    for name, rows in [("fold_metrics.csv", fold_rows), ("class_metrics.csv", class_rows)]:
        saved_rows = pd.read_csv(SNAP / "analysis" / name).to_dict(orient="records")
        assert len(saved_rows) == len(rows)
        for expected, actual in zip(saved_rows, rows):
            for key, value in actual.items():
                assert (abs(float(expected[key]) - float(value)) < 1e-12
                        if isinstance(value, (float, int)) else expected[key] == value), (name, key)
    confusion = np.zeros((18, 18), dtype=int)
    label_index = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(true, single):
        confusion[label_index[actual], label_index[predicted]] += 1
    assert confusion.tolist() == metrics["e28_single_confusion_matrix"]
    wrong_to_right = int(((old != true) & (new == true)).sum())
    right_to_wrong = int(((old == true) & (new != true)).sum())
    assert wrong_to_right == metrics["fusion_wrong_to_right"]
    assert right_to_wrong == metrics["fusion_right_to_wrong"]

    # Each official development prompt has exactly one response per label, so
    # prompt-paired BA deltas equal the mean of 18 hit changes per sampled prompt.
    per_prompt = pd.DataFrame({"prompt_id": e0.prompt_id,
                               "delta": (new == true).astype(int) - (old == true).astype(int)})
    prompt_delta = per_prompt.groupby("prompt_id", sort=True).delta.sum().to_numpy()
    choices = np.random.default_rng(42).integers(0, 120, size=(20_000, 120))
    draws = prompt_delta[choices].sum(axis=1) / 2160
    ci = np.quantile(draws, [.025, .975])
    saved_ci = metrics["paired_prompt_bootstrap_e24_plus_e28_minus_e24"]["confidence_interval_95"]
    assert np.allclose(ci, saved_ci, atol=1e-12)

    delta = scores["e24_plus_e28_fixed"] - scores["e24_fixed"]
    gate = cfg["promotion_gate"]
    gate_results = {
        "delta_at_least_0_01": delta >= gate["min_balanced_accuracy_delta"] - 1e-12,
        "at_least_four_folds_non_decreasing": sum(row["fusion_delta"] >= -1e-12 for row in fold_rows)
        >= gate["min_non_decreasing_folds"],
        "no_class_recall_drop_at_least_0_05": min(row["fusion_delta"] for row in class_rows)
        > -gate["max_allowed_class_recall_drop"] + 1e-12,
    }
    assert gate_results == metrics["gate"]
    assert all(gate_results.values()) == metrics["gate_passed_against_e24"]
    assert not metrics["holdout_evaluated"] and not metrics["fusion_weight_tuned"]

    result = {
        "scope": "independent read-only audit of E28 official development OOF; no holdout/test rows",
        "protocol_lock_sha256": digest(SNAP / "protocol_lock.json"),
        "e28_summary_oof_sha256": digest(SNAP / "analysis/oof_predictions.csv"),
        "e24_reference_oof_sha256": digest(sources["e24_oof"]),
        "protocol_code_config_hashes_match": True,
        "e24_source_hashes_match": True,
        "fold_oof_hashes_and_metadata_match": True,
        "fold_validation_prompts_disjoint_and_match_e0": True,
        "rows": 2160, "prompts": 120, "classes": 18,
        "fold_resources": run_rows,
        "total_train_seconds": sum(row["train_seconds"] for row in run_rows),
        "max_fold_train_reserved_bytes": max(row["train_peak_reserved_bytes"] for row in run_rows),
        "max_fold_validation_reserved_bytes": max(row["validation_peak_reserved_bytes"] for row in run_rows),
        "scores": scores,
        "folds": fold_rows,
        "classes_recall": class_rows,
        "fusion_delta": delta,
        "wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong,
        "paired_prompt_bootstrap": {"repeats": 20_000, "seed": 42, "ci_95": ci.tolist()},
        "gate": gate_results, "gate_passed_against_e24": all(gate_results.values()),
        "resource_gate_passed": resource["gate_passed"],
        "kaggle_hardware_equivalence_proven": resource["kaggle_hardware_equivalence_proven"],
        "server_release_success": release["e28_success"],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["rows", "prompts", "scores", "fusion_delta",
                                                  "folds", "paired_prompt_bootstrap", "gate"]}, indent=2))


if __name__ == "__main__":
    main()
