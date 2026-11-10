#!/usr/bin/env python3
"""Independently check the E24 package, submission, and E0c-only runtime."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle.submit_e24_external_e0c import load_e24_surface
from run_e0 import LABELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("outputs/e24_full_e0c_v1"))
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--submission", type=Path, default=Path("outputs/e24_smoke_submission.parquet"))
    args = parser.parse_args()
    surface, positions = load_e24_surface(args.package, Path(__file__).resolve().parents[1])
    official_text = pd.read_csv(args.train, usecols=["text"], dtype=str, keep_default_na=False)["text"]
    fixture = official_text.iloc[np.arange(4000) % len(official_text)].tolist()
    start = time.perf_counter()
    logits = surface.decision_function(fixture)[:, positions]
    elapsed = time.perf_counter() - start
    if logits.shape != (4000, len(LABELS)) or not np.isfinite(logits).all():
        raise RuntimeError("E24 E0c 4000-row score array is invalid")
    sample = pd.read_csv(args.test, usecols=["id", "text"], dtype=str, keep_default_na=False)
    submission = pd.read_parquet(args.submission)
    if (list(submission.columns) != ["id", "model"] or len(submission) != len(sample)
            or not submission["id"].astype(str).equals(sample["id"])
            or submission["id"].duplicated().any()
            or not set(submission["model"]).issubset(LABELS)):
        raise RuntimeError("E24 submission schema, IDs, or labels invalid")
    print({
        "package_sha256": sha256(args.package / "e0c.joblib"),
        "submission_sha256": sha256(args.submission),
        "sample_rows": len(submission),
        "score_shape": logits.shape,
        "e24_e0c_4000_seconds": round(elapsed, 3),
        "schema_id_order_labels_passed": True,
    })


if __name__ == "__main__":
    main()
