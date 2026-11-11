#!/usr/bin/env python3
"""Freeze, fit, and verify one full-data E30-A adapter after independent OOF audit.

This deliberately imports the locked E30 model, optimizer, data loader, and save/
reload functions. It does not modify the E30 v3 training or validation protocol.
The promotion record is a machine-checkable statement of an independently
audited gate pass. No held-out or test accuracy is computed here.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_e30_supervised_embedding8b as e30
from run_e0 import LABELS, load_and_validate, sha256


PROMOTION_DECISION = "e30_a_promoted_against_strongest_prior_fixed_candidate"
SMOKE_TEXTS = (
    "A concise answer with a few ordinary words.",
    "The result has two parts: first a summary, then an example.",
    "这是一个用于离线重载检查的短句。",
)
SMOKE_ATOL = 2e-3
ARTIFACT_FILES = (
    "head.safetensors", "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
ANALYSIS_FILES = (
    "analysis/metrics.json",
    "analysis/fold_metrics.csv",
    "analysis/class_metrics.csv",
    "analysis/oof_predictions.csv",
)


def artifact_hashes(directory: Path) -> dict[str, str]:
    paths = [path for path in directory.rglob("*") if path.is_file()]
    result = {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(paths)
        if path.relative_to(directory).as_posix()
        not in {"metadata.json", "smoke_reference.json", "reload_report.json"}
    }
    if not set(ARTIFACT_FILES).issubset(result):
        raise ValueError("Saved classifier head is missing")
    return result


def audit_gate(args: argparse.Namespace, cfg: dict) -> dict:
    """Check all frozen evidence before a GPU process or full-data label use."""
    promotion = e30.read_json(args.promotion_record)
    if promotion.get("decision") != PROMOTION_DECISION:
        raise ValueError("Independent audit has not promoted the frozen candidate")
    if promotion.get("approved_fusion") != "e24_plus_e30_fixed_weight_1":
        raise ValueError("Promotion record must name the preregistered fixed E24+E30 fusion")
    if promotion.get("comparison_policy") != "strongest_prior_fixed_including_e28":
        raise ValueError("Promotion record must use the conservative prior-candidate policy")
    if not args.audit_report.is_file():
        raise ValueError("Independent E30 audit report is absent")
    locked_files = {
        "e30_script_sha256": e30.SCRIPT_PATH,
        "e30_config_sha256": e30.CONFIG_PATH,
        "e30_protocol_lock_sha256": args.e30_dir / "protocol_lock.json",
        "e30_resource_report_sha256": args.e30_dir / "resource_report.json",
        "e30_analysis_metrics_sha256": args.e30_dir / ANALYSIS_FILES[0],
        "e30_fold_metrics_sha256": args.e30_dir / ANALYSIS_FILES[1],
        "e30_class_metrics_sha256": args.e30_dir / ANALYSIS_FILES[2],
        "e30_analysis_oof_sha256": args.e30_dir / ANALYSIS_FILES[3],
        "e28_analysis_oof_sha256": args.e28_dir / "analysis/oof_predictions.csv",
        "e36_analysis_metrics_sha256": args.e36_dir / "metrics.json",
        "independent_audit_sha256": args.audit_report,
    }
    for field, path in locked_files.items():
        if promotion.get(field) != sha256(path):
            raise ValueError(f"Promoted evidence changed or is missing: {field}")
    if promotion.get("official_train_sha256") != cfg["official_train_sha256"]:
        raise ValueError("Promotion record names a different official training file")
    if sha256(args.train) != cfg["official_train_sha256"]:
        raise ValueError("Official training CSV differs from the locked E30 data")
    e30.lock_input(argparse.Namespace(
        train=args.train, e0_dir=args.e0_dir, output_dir=args.e30_dir), cfg)

    resource = e30.read_json(args.e30_dir / "resource_report.json")
    if (not resource.get("gate_passed") or not all(resource.get("gate", {}).values())
            or resource.get("config_sha256") != sha256(e30.CONFIG_PATH)
            or resource.get("protocol_lock_sha256") != sha256(args.e30_dir / "protocol_lock.json")
            or resource.get("model_revision") != cfg["model_revision"]
            or resource.get("deployment_model_count") != 1
            or resource.get("training_rows") != 1728
            or resource.get("unlabeled_inference_rows") != 4000
            or resource.get("validation_accuracy_computed") is not False):
        raise ValueError("E30 resource gate is absent, failed, or inconsistent")

    metrics = e30.read_json(args.e30_dir / "analysis/metrics.json")
    scores = metrics.get("balanced_accuracy", {})
    delta = scores.get("e24_plus_e30_fixed", float("nan")) - scores.get("e24_fixed", float("nan"))
    gate = metrics.get("gate", {})
    wanted = {"delta_at_least_0_01", "at_least_four_folds_non_decreasing",
              "no_class_recall_drop_at_least_0_05"}
    if (metrics.get("experiment") != cfg["experiment"]
            or metrics.get("config_sha256") != sha256(e30.CONFIG_PATH)
            or metrics.get("protocol_lock_sha256") != sha256(args.e30_dir / "protocol_lock.json")
            or metrics.get("rows") != 2160 or metrics.get("prompts") != 120
            or metrics.get("labels") != list(LABELS)
            or metrics.get("holdout_evaluated") is not False
            or metrics.get("fusion_weight_tuned") is not False
            or metrics.get("gate_passed_against_e24") is not True
            or set(gate) != wanted or not all(gate.values())
            or not np.isfinite(delta)
            or abs(delta - metrics.get("fusion_delta", float("nan"))) > 1e-12
            or delta < cfg["promotion_gate"]["min_balanced_accuracy_delta"] - 1e-12
            or metrics.get("folds_non_decreasing", 0) < 4
            or metrics.get("worst_class_recall_delta", -1) <= -0.05 + 1e-12):
        raise ValueError("E30 fixed-fusion OOF failed the frozen promotion gate")

    fold_rows = pd.read_csv(args.e30_dir / "analysis/fold_metrics.csv")
    class_rows = pd.read_csv(args.e30_dir / "analysis/class_metrics.csv")
    if (fold_rows["fold"].tolist() != cfg["folds"]
            or len(class_rows) != len(LABELS)
            or class_rows["model"].tolist() != list(LABELS)
            or int(class_rows["support"].sum()) != 2160):
        raise ValueError("E30 fold or class aggregate schema changed")
    if (int((fold_rows["fusion_delta"] >= -1e-12).sum()) != metrics["folds_non_decreasing"]
            or abs(float(class_rows["fusion_delta"].min()) - metrics["worst_class_recall_delta"]) > 1e-12):
        raise ValueError("E30 fold/class aggregates disagree with metrics")
    e36 = e30.read_json(args.e36_dir / "metrics.json")
    old_names = ("e24_fixed", "e24_plus_e28")
    e36_scores = e36.get("balanced_accuracy", {})
    e36_inputs = e36.get("input_sha256", {})
    if (e36.get("experiment") != "e36_fixed_e24_e28_e30_triple_fusion"
            or e36.get("rows") != 2160 or e36.get("prompts") != 120
            or e36.get("labels") != list(LABELS)
            or any(not np.isfinite(e36_scores.get(name, float("nan")))
                   for name in ("e24_fixed", "e24_plus_e28", "e24_plus_e30"))
            or e36.get("holdout_evaluated") is not False
            or e36.get("fusion_weight_or_threshold_tuned") is not False
            or e36_inputs.get("e30_oof") != sha256(args.e30_dir / "analysis/oof_predictions.csv")
            or e36_inputs.get("e30_metrics") != sha256(args.e30_dir / "analysis/metrics.json")
            or e36_inputs.get("e28_oof") != sha256(args.e28_dir / "analysis/oof_predictions.csv")
            or abs(e36_scores.get("e24_fixed", float("nan")) - scores["e24_fixed"]) > 1e-12
            or abs(e36_scores.get("e24_plus_e30", float("nan"))
                   - scores["e24_plus_e30_fixed"]) > 1e-12):
        raise ValueError("E36 fixed-comparator audit does not reproduce E30")
    strongest_old = max(old_names, key=lambda name: e36_scores[name])
    e36_folds = e36.get("fold_metrics", [])
    e36_classes = e36.get("class_metrics", [])
    if (len(e36_folds) != 5 or [row.get("fold") for row in e36_folds] != cfg["folds"]
            or len(e36_classes) != len(LABELS)
            or [row.get("model") for row in e36_classes] != list(LABELS)):
        raise ValueError("E36 fold/class comparison tables changed")
    prior_delta = e36_scores["e24_plus_e30"] - e36_scores[strongest_old]
    prior_nondecrease = sum(row["e24_plus_e30"] >= row[strongest_old] - 1e-12
                            for row in e36_folds)
    prior_worst_class = min(row["e24_plus_e30_recall"] - row[f"{strongest_old}_recall"]
                            for row in e36_classes)
    if (prior_delta < 0.01 - 1e-12 or prior_nondecrease < 4
            or prior_worst_class <= -0.05 + 1e-12):
        raise ValueError(
            "E30 does not pass against the strongest prior fixed candidate: "
            f"{strongest_old}, delta={prior_delta:.6f}, folds={prior_nondecrease}, "
            f"worst class={prior_worst_class:.6f}")
    if promotion.get("strongest_prior_fixed_candidate") != strongest_old:
        raise ValueError("Promotion record names a different strongest prior comparator")
    for fold in cfg["folds"]:
        fold_dir = args.e30_dir / f"fold_{fold}"
        record = e30.read_json(fold_dir / "run.json")
        if (record.get("fold") != fold or record.get("epochs") != cfg["epochs"]
                or record.get("train_rows") != 1728 or record.get("validation_rows") != 432
                or record.get("config_sha256") != sha256(e30.CONFIG_PATH)
                or record.get("protocol_lock_sha256") != sha256(args.e30_dir / "protocol_lock.json")
                or record.get("resource_report_sha256") != sha256(args.e30_dir / "resource_report.json")
                or record.get("model_revision") != cfg["model_revision"]
                or record.get("model_file_sha256") != resource["model_file_sha256"]
                or record.get("oof_predictions_sha256") != sha256(fold_dir / "oof_predictions.csv")
                or record.get("no_epoch_or_threshold_selection") is not True
                or record.get("holdout_evaluated") is not False):
            raise ValueError(f"E30 fold {fold} is missing or differs from the audit")
    return {"promotion_record_sha256": sha256(args.promotion_record),
            "analysis_metrics_sha256": sha256(args.e30_dir / "analysis/metrics.json"),
            "resource_report_sha256": sha256(args.e30_dir / "resource_report.json"),
            "independent_audit_sha256": sha256(args.audit_report),
            "fusion_delta": delta, "strongest_prior_fixed_candidate": strongest_old,
            "delta_vs_strongest_prior_fixed": prior_delta,
            "folds_nondecreasing_vs_strongest_prior_fixed": prior_nondecrease,
            "worst_class_delta_vs_strongest_prior_fixed": prior_worst_class}


def check(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    cfg = e30.config()
    if (cfg["seed"] != 42 or cfg["epochs"] != 3 or cfg["max_length"] != 256
            or cfg["deployment_model_count"] != 1
            or cfg["input"] != "raw_answer_text_with_official_tokenizer_special_tokens_no_chat_template"
            or cfg["pooling"] != "last_non_padding_token_last_hidden_layer"
            or cfg["loss"] != "unweighted_cross_entropy"):
        raise ValueError("Full fit recipe differs from the audited E30-A recipe")
    evidence = audit_gate(args, cfg)
    model_hashes = e30.verify_model_dir(args.model_dir, cfg)
    resource = e30.read_json(args.e30_dir / "resource_report.json")
    if resource.get("model_file_sha256") != model_hashes:
        raise ValueError("Full-fit base model differs from the E30 resource run")
    official = load_and_validate(args.train)
    if (len(official) != 2699 or official["prompt_id"].nunique() != 150
            or set(official["model"]) != set(LABELS)):
        raise ValueError("Full official training dimensions changed")
    return cfg, evidence, model_hashes


def smoke_scores(model, tokenizer, cfg: dict) -> np.ndarray:
    loader = e30.make_loader(list(SMOKE_TEXTS), None, tokenizer, cfg, shuffle=False)
    scores = e30.predict(model, loader)
    if scores.shape != (len(SMOKE_TEXTS), len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError("E30 smoke logits are invalid")
    return scores


def fit(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise ValueError(f"Full-fit output already exists: {args.output_dir}")
    cfg, evidence, model_hashes = check(args)
    versions = e30.runtime_versions(cfg)
    if versions != e30.read_json(args.e30_dir / "resource_report.json").get("packages"):
        raise ValueError("Full-fit runtime differs from the audited E30 resource run")
    e30.gpu_preflight(cfg)
    import torch

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    official = load_and_validate(args.train)
    tokenizer = e30.load_tokenizer(args.model_dir)
    targets = {label: position for position, label in enumerate(LABELS)}
    loader = e30.make_loader(official["text"].tolist(),
                             [targets[label] for label in official["model"]],
                             tokenizer, cfg, shuffle=True)
    e30.reset_cuda_peak_stats()
    model = e30.build_model(args.model_dir, cfg)
    train_report = e30.train(model, loader, cfg, epochs=cfg["epochs"])
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    if train_peak_reserved >= cfg["resource_gate"]["max_train_peak_gib"] * 2**30:
        raise RuntimeError("Full-fit training peak exceeded the preregistered E30 memory cap")
    reference = smoke_scores(model, tokenizer, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    e30.save_adapter_head(model, tokenizer, args.output_dir)
    e30.write_new_json(args.output_dir / "smoke_reference.json", {
        "texts": list(SMOKE_TEXTS), "labels": list(LABELS),
        "logits": reference.tolist(), "atol": SMOKE_ATOL,
    })
    metadata = {
        "recipe": "e30_a_full_official_after_independent_oof_audit",
        "source_script_sha256": sha256(Path(__file__)),
        "locked_e30_script_sha256": sha256(e30.SCRIPT_PATH),
        "locked_e30_config_sha256": sha256(e30.CONFIG_PATH),
        "official_train_sha256": sha256(args.train),
        "training_rows": len(official), "training_prompts": int(official["prompt_id"].nunique()),
        "labels": list(LABELS), "model_revision": cfg["model_revision"],
        "model_file_sha256": model_hashes, "packages": versions,
        "seed": cfg["seed"], "epochs": cfg["epochs"],
        "max_length": cfg["max_length"], "deployment_model_count": 1,
        **train_report,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "audit_evidence": evidence,
        "artifact_sha256": artifact_hashes(args.output_dir),
        "smoke_reference_sha256": sha256(args.output_dir / "smoke_reference.json"),
        "holdout_evaluated": False, "test_evaluated": False,
        "full_official_fit_after_candidate_freeze": True,
    }
    e30.write_new_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps({"training_rows": len(official), "training_prompts": metadata["training_prompts"],
                      "training_seconds": train_report["train_seconds"],
                      "train_peak_reserved_bytes": train_peak_reserved,
                      "artifact_files": len(metadata["artifact_sha256"]),
                      "metadata_sha256": sha256(args.output_dir / "metadata.json")}, indent=2))


def verify_reload(args: argparse.Namespace) -> None:
    if (args.output_dir / "reload_report.json").exists():
        raise ValueError("Reload report already exists; refusing to overwrite it")
    cfg, evidence, model_hashes = check(args)
    metadata = e30.read_json(args.output_dir / "metadata.json")
    reference = e30.read_json(args.output_dir / "smoke_reference.json")
    if (metadata.get("recipe") != "e30_a_full_official_after_independent_oof_audit"
            or metadata.get("source_script_sha256") != sha256(Path(__file__))
            or metadata.get("official_train_sha256") != sha256(args.train)
            or metadata.get("locked_e30_script_sha256") != sha256(e30.SCRIPT_PATH)
            or metadata.get("locked_e30_config_sha256") != sha256(e30.CONFIG_PATH)
            or metadata.get("model_file_sha256") != model_hashes
            or metadata.get("audit_evidence") != evidence
            or metadata.get("labels") != list(LABELS)
            or metadata.get("training_rows") != 2699
            or metadata.get("training_prompts") != 150
            or metadata.get("seed") != 42 or metadata.get("epochs") != 3
            or metadata.get("max_length") != 256
            or metadata.get("artifact_sha256") != artifact_hashes(args.output_dir)
            or metadata.get("smoke_reference_sha256") != sha256(args.output_dir / "smoke_reference.json")
            or reference.get("texts") != list(SMOKE_TEXTS)
            or reference.get("labels") != list(LABELS)
            or reference.get("atol") != SMOKE_ATOL):
        raise ValueError("Full-fit package or promotion evidence changed")
    versions = e30.runtime_versions(cfg)
    if metadata.get("packages") != versions:
        raise ValueError("Reload runtime differs from the fitted package")
    e30.gpu_preflight(cfg)
    import torch

    tokenizer = e30.load_tokenizer(args.output_dir / "tokenizer")
    e30.reset_cuda_peak_stats()
    started = time.perf_counter()
    model = e30.reload_model(args.model_dir, args.output_dir, cfg)
    reload_seconds = time.perf_counter() - started
    actual = smoke_scores(model, tokenizer, cfg)
    expected = np.asarray(reference["logits"], dtype=np.float64)
    if (expected.shape != actual.shape
            or not np.allclose(actual, expected, rtol=0, atol=SMOKE_ATOL)
            or not np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))):
        raise ValueError("Full E30 adapter reload changed the fixed smoke scores")
    official = load_and_validate(args.train)
    source = official["text"].tolist()
    fixture = [source[index % len(source)] for index in range(4000)]
    loader = e30.make_loader(fixture, None, tokenizer, cfg, shuffle=False)
    started = time.perf_counter()
    scores = e30.predict(model, loader)
    inference_seconds = time.perf_counter() - started
    if scores.shape != (4000, len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError("Full E30 4,000-row reload inference failed")
    peak_reserved = torch.cuda.max_memory_reserved(0)
    if (peak_reserved >= cfg["resource_gate"]["max_train_peak_gib"] * 2**30
            or inference_seconds > cfg["resource_gate"]["max_four_thousand_inference_seconds"]):
        raise RuntimeError("Full E30 reload inference exceeded resource limits")
    del model, scores
    gc.collect()
    e30.write_new_json(args.output_dir / "reload_report.json", {
        "recipe": metadata["recipe"], "metadata_sha256": sha256(args.output_dir / "metadata.json"),
        "model_revision": cfg["model_revision"], "training_rows": 2699,
        "unlabeled_inference_rows": 4000, "adapter_reload_seconds": reload_seconds,
        "inference_seconds": inference_seconds, "peak_reserved_bytes": peak_reserved,
        "smoke_max_abs_delta": float(np.max(np.abs(actual - expected))),
        "smoke_argmax_equal": True, "finite_four_thousand_logits": True,
        "holdout_accuracy_computed": False, "test_accuracy_computed": False,
        "gate_passed": True,
    })
    print(json.dumps({"gate_passed": True, "inference_seconds": inference_seconds,
                      "peak_reserved_bytes": peak_reserved,
                      "reload_report_sha256": sha256(args.output_dir / "reload_report.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "fit", "verify-reload"))
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--e30-dir", type=Path, default=Path("outputs/e30_qwen3_embedding8b_v3"))
    parser.add_argument("--e28-dir", type=Path, default=Path("outputs/e28_qwen8b_fivefold"))
    parser.add_argument("--e36-dir", type=Path, default=Path("outputs/e36_fixed_triple_fusion"))
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local pinned Qwen3-Embedding-8B snapshot, no download")
    parser.add_argument("--audit-report", type=Path, required=True,
                        help="Independent audit report whose SHA-256 is in promotion record")
    parser.add_argument("--promotion-record", type=Path, required=True,
                        help="Machine-checkable post-audit promotion record; required before GPU")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e30_full_official_v1"))
    args = parser.parse_args()
    if args.stage == "check":
        cfg, evidence, hashes = check(args)
        print(json.dumps({"gate_passed": True, "experiment": cfg["experiment"],
                          "candidate_fusion": "e24_plus_e30_fixed_weight_1",
                          "model_revision": cfg["model_revision"],
                          "verified_model_files": len(hashes), "evidence": evidence}, indent=2))
    elif args.stage == "fit":
        fit(args)
    else:
        verify_reload(args)


if __name__ == "__main__":
    main()
