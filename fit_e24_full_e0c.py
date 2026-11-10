#!/usr/bin/env python3
"""Fit the locked E24 external-data E0c replacement on all official rows.

This does not refit RoBERTa or ModernBERT. The external selection is exactly
the audited E24 fivefold selection; only per-class sample weights change to
keep the external/official loss-weight ratio at 0.25 with 149/150 full rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from run_combined import make_pipeline
from run_e0 import LABELS, load_and_validate, sha256
from run_e24_prvsiyan_external_ablation import (
    EXPECTED_TRAIN_SHA256,
    select_external,
)


SELECTION_SHA256 = "d7f8b7c9b0480ec896777e667589bf7d2fe12e503aa8f867c0708d0656ad6191"
PROMPTS_SHA256 = "2b917303a82218ba36a11cf61288276d72cf7d8d19f0feb914eb1434673ed266"
FIT_METRICS_SHA256 = "ce4b4cbce06539e054f0bc1702d29f22a80dbfbdb4366eb51ffa805ecfccc47f"
RATIO = 0.25
RECIPE = "e24_audited_prvsiyan_v9_full_e0c_replacement"
SOURCE_FILES = (
    "fit_e24_full_e0c.py", "run_e24_prvsiyan_external_ablation.py",
    "run_e0.py", "run_style.py", "run_combined.py",
)
SMOKE_TEXTS = (
    "The result follows from the three observations in the question.",
    "First, consider a counterexample; then revise the explanation.",
    "A short response can still be precise and helpful.",
)


def validate_locked_selection(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    for path, expected in (
        (args.selected_manifest, SELECTION_SHA256),
        (args.selected_prompts, PROMPTS_SHA256),
        (args.fit_metrics, FIT_METRICS_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Locked E24 fold output changed: {path}")
    metrics = json.loads(args.fit_metrics.read_text(encoding="utf-8"))
    if (metrics.get("selection", {}).get("n_selected_external_rows") != 3620
            or metrics.get("n_folds") != 5
            or metrics.get("original_e0c_ba") != 0.34259259259259256):
        raise ValueError("E24 fit record differs from the frozen fivefold run")
    selected, prompts, selection_meta = select_external(
        args.source, args.audit_manifest, args.audit_report)
    if len(selected) != 3620 or selected["model"].nunique() != 17:
        raise ValueError("E24 selected external row count/classes changed")
    locked = pd.read_csv(args.selected_manifest, dtype=str, keep_default_na=False)
    columns = [
        "id", "prompt_id", "model", "served_model_id", "served_provider",
        "provider_route",
    ]
    if list(locked.columns) != columns + ["sample_weight"]:
        raise ValueError("Unexpected selected manifest schema")
    for key in columns:
        if not np.array_equal(locked[key].to_numpy(), selected[key].to_numpy()):
            raise ValueError(f"Selected external {key} differs from frozen manifest")
    if not np.allclose(
        locked["sample_weight"].to_numpy(dtype=np.float64),
        selected["sample_weight"].to_numpy(dtype=np.float64), rtol=0, atol=1e-12,
    ):
        raise ValueError("Selected external fold weights differ from frozen manifest")
    prompt_manifest = pd.read_csv(args.selected_prompts, dtype=str, keep_default_na=False)
    if (not prompt_manifest["prompt_id"].equals(prompts["prompt_id"])
            or not prompt_manifest["hash"].equals(prompts["hash"])
            or not prompt_manifest["selected"].str.lower().equals(
                prompts["selected"].astype(str).str.lower()
            )):
        raise ValueError("Selected prompt hashes differ from frozen manifest")
    return selected, selection_meta


def fit(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    if sklearn.__version__ != "1.7.2" or sys.version_info[:3] != (3, 10, 12):
        raise ValueError("Fit requires original Python 3.10.12 / scikit-learn 1.7.2")
    selected, selection_meta = validate_locked_selection(args)
    if sha256(args.train) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Official train.csv differs from frozen E24/E16 data")
    official = load_and_validate(args.train)
    if len(official) != 2699 or official["prompt_id"].nunique() != 150:
        raise ValueError("Expected all 2699 official rows and 150 prompts")
    official_counts = official["model"].value_counts().to_dict()
    external_counts = selected["model"].value_counts().to_dict()
    if any(official_counts.get(label) not in (149, 150) for label in LABELS):
        raise ValueError("Unexpected official class count")
    if {label for label, count in official_counts.items() if count == 149} != {"claude-fable-5"}:
        raise ValueError("Unexpected missing official response")
    if set(external_counts) != set(LABELS) - {"qwen3.8-max"}:
        raise ValueError("Expected 17 audited external classes and no Qwen")
    weights_by_class = {
        label: RATIO * official_counts[label] / external_counts[label]
        for label in external_counts
    }
    external_weight = selected["model"].map(weights_by_class).to_numpy(dtype=np.float64)
    for label, count in external_counts.items():
        total = float(external_weight[selected["model"].to_numpy() == label].sum())
        if not np.isclose(total, RATIO * official_counts[label], atol=1e-10):
            raise AssertionError(f"External class weight ratio failed for {label}")
    texts = np.concatenate((official["text"].to_numpy(), selected["text"].to_numpy()))
    labels = np.concatenate((official["model"].to_numpy(), selected["model"].to_numpy()))
    weights = np.concatenate((np.ones(len(official), dtype=np.float64), external_weight))
    start = time.perf_counter()
    model = make_pipeline()
    model.fit(texts, labels, classifier__sample_weight=weights)
    elapsed = time.perf_counter() - start
    classes = model.named_steps["classifier"].classes_
    if set(classes) != set(LABELS):
        raise ValueError("Full E24 E0c model lost an official class")
    positions = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    smoke = model.decision_function(list(SMOKE_TEXTS))[:, positions]
    if smoke.shape != (3, len(LABELS)) or not np.isfinite(smoke).all():
        raise ValueError("Full E24 E0c smoke logits are invalid")
    args.output_dir.mkdir(parents=True)
    joblib.dump(model, args.output_dir / "e0c.joblib", compress=3)
    smoke_ref = {"texts": list(SMOKE_TEXTS), "labels": list(LABELS), "logits": smoke.tolist()}
    (args.output_dir / "smoke_reference.json").write_text(
        json.dumps(smoke_ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    root = Path(__file__).resolve().parent
    metadata = {
        "package_schema_version": 1,
        "recipe": RECIPE,
        "train_sha256": sha256(args.train),
        "n_official_rows": len(official),
        "n_official_prompts": int(official["prompt_id"].nunique()),
        "n_external_rows": len(selected),
        "n_external_prompts": int(selected["prompt_id"].nunique()),
        "labels": list(LABELS),
        "official_rows_by_class": {label: int(official_counts[label]) for label in LABELS},
        "external_rows_by_class": {label: int(external_counts.get(label, 0)) for label in LABELS},
        "external_total_weight_by_class": {
            label: float(RATIO * official_counts[label]) if label in external_counts else 0.0
            for label in LABELS
        },
        "external_to_official_weight_ratio": RATIO,
        "selection": selection_meta,
        "selected_manifest_sha256": SELECTION_SHA256,
        "selected_prompts_sha256": PROMPTS_SHA256,
        "e24_fit_metrics_sha256": FIT_METRICS_SHA256,
        "source_sha256": {name: sha256(root / name) for name in SOURCE_FILES},
        "artifact_sha256": {
            "e0c.joblib": sha256(args.output_dir / "e0c.joblib"),
            "smoke_reference.json": sha256(args.output_dir / "smoke_reference.json"),
        },
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "fit_seconds": elapsed,
        "official_labels_used_for_full_fit_only": True,
        "external_source_protocol_used_global_official_length_statistics": True,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir), "fit_seconds": elapsed,
        "model_sha256": metadata["artifact_sha256"]["e0c.joblib"],
        "official_rows": len(official), "external_rows": len(selected),
        "external_weight_ratio": RATIO,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/external/prvsiyan/passages.csv"))
    parser.add_argument("--audit-manifest", type=Path,
                        default=Path("data/external/prvsiyan/e24_accepted_manifest.csv"))
    parser.add_argument("--audit-report", type=Path,
                        default=Path("outputs/e24_prvsiyan_audit/report.json"))
    parser.add_argument("--selected-manifest", type=Path,
                        default=Path("outputs/e24_prvsiyan_fit/selected_external.csv"))
    parser.add_argument("--selected-prompts", type=Path,
                        default=Path("outputs/e24_prvsiyan_fit/selected_prompt_ids.csv"))
    parser.add_argument("--fit-metrics", type=Path,
                        default=Path("outputs/e24_prvsiyan_fit/fit_metrics.json"))
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e24_full_e0c_v1"))
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
