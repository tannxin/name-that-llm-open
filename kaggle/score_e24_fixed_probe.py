#!/usr/bin/env python3
"""Export locked E24 family scores for an unscored, text-only deployment proxy."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e39_probe_common as common  # noqa: E402
import submit_e24_external_e0c as e24  # noqa: E402


def score(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    if list(e24.MODERN_LABELS) != list(e24.LABELS):
        raise ValueError("E24 and ModernBERT label orders differ")
    paths = (args.modern_seed42, args.modern_seed17, args.modern_seed73)
    resolved = tuple(path.resolve(strict=True) for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("ModernBERT seeds must use separate packages")
    for seed, path in zip(e24.EXPECTED_SEEDS, resolved, strict=True):
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        e24.validate_modern_metadata(path, metadata, seed)
    old_metadata = json.loads((args.full_package / "metadata.json").read_text(encoding="utf-8"))
    for key, expected in (("train_sha256", common.TRAIN_SHA256),
                          ("n_training_rows", 2699), ("n_training_prompts", 150)):
        if old_metadata.get(key) != expected:
            raise ValueError(f"E24 base package has unexpected {key}")

    ids, texts = common.proxy_fixture(args.train)
    n_rows = len(texts)
    surface, positions = e24.load_e24_surface(args.e24_package,
                                               Path(__file__).resolve().parents[1])
    e24_logits = np.asarray(surface.decision_function(texts)[:, positions], dtype=np.float64)
    _, original_e0c, roberta = e24.predict_package(args.full_package, texts, "full")
    if original_e0c.shape != e24_logits.shape:
        raise RuntimeError("E24 E0c and original E0c shapes differ")
    family = e24.row_standardize(e24_logits, n_rows)
    family += e24.row_standardize(roberta, n_rows)
    del surface, e24_logits, original_e0c, roberta
    gc.collect()

    modern = np.zeros_like(family)
    for seed, path in zip(e24.EXPECTED_SEEDS, resolved, strict=True):
        model, tokenizer, metadata = e24.load_modern_package(path, device="cpu")
        e24.validate_modern_metadata(path, metadata, seed)
        smoke = json.loads((path / "smoke_reference.json").read_text(encoding="utf-8"))
        if smoke.get("texts") != list(e24.SMOKE_TEXTS):
            raise RuntimeError(f"ModernBERT seed {seed} smoke texts changed")
        expected = np.asarray(smoke.get("logits"), dtype=np.float32)
        actual = e24.score_modern_texts(model, tokenizer, list(e24.SMOKE_TEXTS),
                                        batch_size=2, device="cpu")
        if (expected.shape != actual.shape or
                not np.allclose(expected, actual, rtol=0,
                                atol=e24.CROSS_RUNTIME_SMOKE_ATOL) or
                not np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))):
            raise RuntimeError(f"ModernBERT seed {seed} failed reload smoke")
        logits = e24.score_modern_texts(model, tokenizer, texts,
                                        batch_size=args.modern_batch_size, device="cpu")
        modern += e24.row_standardize(logits, n_rows) / len(e24.EXPECTED_SEEDS)
        del model, tokenizer, logits
        gc.collect()

    combined = family + modern
    if combined.shape != (common.PROXY_ROWS, len(e24.LABELS)) or not np.isfinite(combined).all():
        raise RuntimeError("E24 fixed score matrix is invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids, labels=np.asarray(e24.LABELS, dtype=str),
                        scores=combined)
    print(f"E24_FIXED_PROXY_SCORES_READY rows={len(ids)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-package", type=Path, required=True)
    parser.add_argument("--modern-seed42", type=Path, required=True)
    parser.add_argument("--modern-seed17", type=Path, required=True)
    parser.add_argument("--modern-seed73", type=Path, required=True)
    parser.add_argument("--e24-package", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modern-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.modern_batch_size < 1:
        raise ValueError("Batch size must be positive")
    score(args)


if __name__ == "__main__":
    main()
