#!/usr/bin/env python3
"""Offline submission using the validated full package and fixed z-score fusion.

This changes only the fusion rule. The fitted E0c and RoBERTa weights, source
hashes, label order, and official test validation remain in final_pipeline.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_pipeline import load_test, predict_package  # noqa: E402
from run_e0 import LABELS  # noqa: E402


def row_standardize(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if not np.isfinite(scores).all() or (scale <= 0).any():
        raise ValueError("Non-finite or constant class scores")
    return centered / scale


def submit(full_package: Path, test_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise ValueError(f"Submission already exists: {output_path}")
    test = load_test(test_path)
    _, surface, neural = predict_package(full_package, test["text"].tolist(), "full")
    if surface.shape != neural.shape or surface.shape != (len(test), len(LABELS)):
        raise ValueError("Unexpected fitted-model score shape")
    combined = row_standardize(surface) + row_standardize(neural)
    predicted = np.asarray(LABELS, dtype=object)[combined.argmax(axis=1)]
    output = pd.DataFrame({"id": test["id"], "model": predicted})
    if not set(output["model"]).issubset(LABELS):
        raise ValueError("Unknown predicted label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False, engine="pyarrow")
    if not pd.read_parquet(output_path, engine="pyarrow").equals(output):
        raise RuntimeError("Parquet round trip changed IDs or predictions")
    print(f"Offline z-score submission written: {output_path} ({len(output)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-package", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    submit(args.full_package, args.test, args.output)


if __name__ == "__main__":
    main()
