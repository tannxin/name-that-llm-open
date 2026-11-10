#!/usr/bin/env python3
"""Reproducible CPU inference timing using development text only, without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SPLIT_SHA256 = "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def development_texts(train_path: Path, split_path: Path, n_rows: int) -> pd.DataFrame:
    if sha256(train_path) != TRAIN_SHA256 or sha256(split_path) != SPLIT_SHA256:
        raise ValueError("Training text or fixed split manifest hash changed")
    frame = pd.read_csv(train_path, usecols=["prompt_id", "text"], dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    if len(frame) != 2699 or len(split) != 150 or set(split.columns) != {"prompt_id", "split"}:
        raise ValueError("Unexpected training rows or split manifest")
    if split["prompt_id"].duplicated().any() or not set(split["split"]) <= {"development", "holdout"}:
        raise ValueError("Invalid prompt split manifest")
    selected = frame.merge(split, on="prompt_id", validate="many_to_one")
    selected = selected.loc[selected["split"].eq("development")]
    if len(selected) != 2160 or selected["prompt_id"].nunique() != 120:
        raise ValueError("Expected 2,160 development rows from 120 prompts")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(selected), size=n_rows)
    sampled = selected.iloc[indices]
    return pd.DataFrame({
        "id": [f"benchmark-{index:05d}" for index in range(n_rows)],
        "text": sampled["text"].tolist(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/e0c_fixed_grid/split_manifest.csv"))
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--write-test", type=Path)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write an unlabeled benchmark CSV without running a model")
    args = parser.parse_args()
    if args.rows < 1 or args.batch_size < 1:
        raise ValueError("Rows and batch size must be positive")
    test = development_texts(args.train, args.split, args.rows)
    if args.write_test is not None:
        if args.write_test.exists():
            raise ValueError(f"Benchmark test already exists: {args.write_test}")
        args.write_test.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(args.write_test, index=False)
    if args.prepare_only:
        if args.write_test is None:
            raise ValueError("--prepare-only requires --write-test")
        print(json.dumps({
            "rows": args.rows,
            "source": "fixed development prompt texts; labels not read",
            "test_path": str(args.write_test),
            "test_sha256": sha256(args.write_test),
        }, indent=2), flush=True)
        return
    if args.package is None:
        raise ValueError("--package is required unless --prepare-only is set")
    from fit_modernbert_full import LABELS, load_package, score_texts
    started = time.perf_counter()
    model, tokenizer, metadata = load_package(args.package, device="cpu")
    loaded = time.perf_counter()
    scores = score_texts(model, tokenizer, test["text"].tolist(),
                         batch_size=args.batch_size, device="cpu")
    scored = time.perf_counter()
    if scores.shape != (args.rows, len(LABELS)) or not np.isfinite(scores).all():
        raise RuntimeError("CPU benchmark returned invalid class scores")
    result = {
        "seed": metadata["seed"],
        "rows": args.rows,
        "batch_size": args.batch_size,
        "device": "cpu",
        "source": "fixed development prompt texts, sampled with replacement; labels not read",
        "load_seconds": loaded - started,
        "score_seconds": scored - loaded,
        "total_seconds": scored - started,
        "rows_per_second": args.rows / (scored - loaded),
        "projected_score_seconds_4000": 4000 * (scored - loaded) / args.rows,
        "test_sha256": sha256(args.write_test) if args.write_test is not None else None,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
