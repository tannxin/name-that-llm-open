#!/usr/bin/env python3
"""Offline E16 submission: E0c, RoBERTa, and a three-seed ModernBERT stream.

Every component is standardized across its 18 class scores within each row.
The ModernBERT stream is the mean of three separately standardized seeds;
the final score is the equal-weight sum of the three model-family streams.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_pipeline import load_test, predict_package  # noqa: E402
from fit_modernbert_full import (  # noqa: E402
    LABELS as MODERN_LABELS,
    SMOKE_TEXTS,
    load_package as load_modern_package,
    score_texts as score_modern_texts,
)
from run_e0 import LABELS  # noqa: E402


EXPECTED_SEEDS = (42, 17, 73)
EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_MODEL_ID = "answerdotai/ModernBERT-base"
EXPECTED_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
# Package creation already requires <=2e-5 on the training machine. The
# submission may run on a different CPU build; keep a separate cross-runtime
# smoke tolerance while still requiring identical argmax and all file hashes.
CROSS_RUNTIME_SMOKE_ATOL = 1e-3


def row_standardize(scores: np.ndarray, n_rows: int) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.shape != (n_rows, len(LABELS)) or not np.isfinite(array).all():
        raise ValueError(f"Invalid or non-finite 18-class scores: {array.shape}")
    centered = array - array.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if (scale <= 0).any() or not np.isfinite(scale).all():
        raise ValueError("Constant or non-finite 18-class score row")
    return centered / scale


def validate_modern_metadata(path: Path, metadata: dict, expected_seed: int) -> None:
    if not isinstance(metadata, dict):
        raise ValueError(f"ModernBERT metadata is not an object: {path}")
    required = {
        "package_schema_version": 1,
        "mode": "full",
        "seed": expected_seed,
        "model_id": EXPECTED_MODEL_ID,
        "requested_revision": EXPECTED_REVISION,
        "pretrained_revision": EXPECTED_REVISION,
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "n_training_rows": 2699,
        "n_training_prompts": 150,
        "labels": list(LABELS),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"ModernBERT seed {expected_seed} has unexpected {key}: {path}")
    if not metadata.get("artifact_sha256") or not metadata.get("source_sha256"):
        raise ValueError(f"ModernBERT package hashes missing: {path}")
    config = metadata.get("training_config", {})
    for key, expected in (
        ("epochs", 12), ("max_length", 256), ("batch_size", 16),
        ("dropout", 0.1), ("pooling", "mean"),
    ):
        if config.get(key) != expected:
            raise ValueError(f"ModernBERT seed {expected_seed} has unexpected {key}: {path}")
    runtime = metadata.get("runtime_versions", {})
    if runtime.get("torch", "").split("+")[0] != "2.6.0" or runtime.get("transformers") != "4.51.3":
        raise ValueError(f"ModernBERT seed {expected_seed} runtime differs from fitted runtime")


def submit(
    full_package: Path,
    modern_packages: tuple[Path, Path, Path],
    test_path: Path,
    output_path: Path,
    batch_size: int = 16,
) -> None:
    if output_path.exists():
        raise ValueError(f"Submission already exists: {output_path}")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if len(MODERN_LABELS) != len(LABELS) or list(MODERN_LABELS) != list(LABELS):
        raise ValueError("ModernBERT label order differs from the official fixed label order")
    paths = tuple(path.resolve(strict=True) for path in modern_packages)
    if len(set(paths)) != len(paths):
        raise ValueError("Each ModernBERT seed must use a different package directory")
    for expected_seed, path in zip(EXPECTED_SEEDS, paths, strict=True):
        with (path / "metadata.json").open("r", encoding="utf-8") as handle:
            declared_metadata = json.load(handle)
        validate_modern_metadata(path, declared_metadata, expected_seed)
    old_metadata = json.loads((full_package / "metadata.json").read_text(encoding="utf-8"))
    for key, expected in (
        ("train_sha256", EXPECTED_TRAIN_SHA256),
        ("n_training_rows", 2699),
        ("n_training_prompts", 150),
    ):
        if old_metadata.get(key) != expected:
            raise ValueError(f"E0c/RoBERTa package has unexpected {key}")

    test = load_test(test_path)
    texts = test["text"].tolist()
    n_rows = len(test)
    print(f"Scoring {n_rows} official test rows", flush=True)

    _, surface_logits, roberta_logits = predict_package(full_package, texts, "full")
    family_score = row_standardize(surface_logits, n_rows)
    family_score += row_standardize(roberta_logits, n_rows)
    del surface_logits, roberta_logits
    gc.collect()
    print("E0c and RoBERTa streams complete", flush=True)

    modern_stream = np.zeros_like(family_score)
    for expected_seed, path in zip(EXPECTED_SEEDS, paths, strict=True):
        model, tokenizer, checked_metadata = load_modern_package(path, device="cpu")
        validate_modern_metadata(path, checked_metadata, expected_seed)
        reference = json.loads((path / "smoke_reference.json").read_text(encoding="utf-8"))
        if reference.get("texts") != list(SMOKE_TEXTS):
            raise ValueError(f"ModernBERT seed {expected_seed} smoke texts changed")
        expected_smoke = np.asarray(reference.get("logits"), dtype=np.float32)
        actual_smoke = score_modern_texts(
            model, tokenizer, list(SMOKE_TEXTS), batch_size=2, device="cpu"
        )
        if expected_smoke.shape != actual_smoke.shape or not np.allclose(
            expected_smoke, actual_smoke, rtol=0, atol=CROSS_RUNTIME_SMOKE_ATOL
        ):
            raise RuntimeError(f"ModernBERT seed {expected_seed} failed reload smoke check")
        if not np.array_equal(expected_smoke.argmax(axis=1), actual_smoke.argmax(axis=1)):
            raise RuntimeError(f"ModernBERT seed {expected_seed} changed smoke predictions")
        logits = score_modern_texts(
            model, tokenizer, texts, batch_size=batch_size, device="cpu"
        )
        modern_stream += row_standardize(logits, n_rows) / len(EXPECTED_SEEDS)
        del model, tokenizer, logits
        gc.collect()
        print(f"ModernBERT seed {expected_seed} complete", flush=True)

    combined = family_score + modern_stream
    predictions = np.asarray(LABELS, dtype=object)[combined.argmax(axis=1)]
    output = pd.DataFrame({"id": test["id"], "model": predictions})
    if list(output.columns) != ["id", "model"] or len(output) != n_rows:
        raise RuntimeError("Submission schema or row count changed")
    if not output["id"].equals(test["id"]) or not set(output["model"]).issubset(LABELS):
        raise RuntimeError("Submission IDs or labels are invalid")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"Stale temporary submission exists: {temporary}")
    try:
        output.to_parquet(temporary, index=False, engine="pyarrow")
        roundtrip = pd.read_parquet(temporary, engine="pyarrow")
        if not roundtrip.equals(output):
            raise RuntimeError("Parquet round trip changed IDs or predictions")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"E16 offline submission written: {output_path} ({n_rows} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-package", required=True, type=Path)
    for seed in EXPECTED_SEEDS:
        parser.add_argument(f"--modern-seed{seed}", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modern-batch-size", type=int, default=16)
    args = parser.parse_args()
    submit(
        args.full_package,
        (args.modern_seed42, args.modern_seed17, args.modern_seed73),
        args.test,
        args.output,
        args.modern_batch_size,
    )


if __name__ == "__main__":
    main()
