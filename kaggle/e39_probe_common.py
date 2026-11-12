#!/usr/bin/env python3
"""Shared, label-free inputs and output checks for the E39 Kaggle deployment probe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
PROXY_ROWS = 4000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def train_texts_only(path: Path) -> list[str]:
    if sha256_file(path) != TRAIN_SHA256:
        raise ValueError("Official training CSV hash differs from the locked protocol")
    # The model/label columns are not loaded into the deployment probe.
    frame = pd.read_csv(path, usecols=["text"], dtype={"text": str}, keep_default_na=False,
                        encoding="utf-8-sig")
    texts = frame["text"].tolist()
    if len(texts) != 2699 or any(not text.strip() for text in texts):
        raise ValueError("Unexpected official training text fixture")
    return texts


def proxy_fixture(path: Path) -> tuple[np.ndarray, list[str]]:
    source = train_texts_only(path)
    texts = (source * ((PROXY_ROWS + len(source) - 1) // len(source)))[:PROXY_ROWS]
    ids = np.asarray([f"e39-proxy-{index:05d}" for index in range(PROXY_ROWS)], dtype=str)
    if len(set(ids)) != PROXY_ROWS or len(texts) != PROXY_ROWS:
        raise RuntimeError("Proxy fixture is not exactly 4000 unique rows")
    return ids, texts


def row_z(scores: np.ndarray, labels: tuple[str, ...], n_rows: int) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.shape != (n_rows, len(labels)) or not np.isfinite(array).all():
        raise ValueError(f"Invalid 18-class score matrix: {array.shape}")
    centered = array - array.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Flat or non-finite score row")
    return centered / scale


def write_proxy_submission(path: Path, ids: np.ndarray, e24_scores: np.ndarray,
                           e39_logits: np.ndarray, labels: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"Probe output already exists: {path}")
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError("Official label order is not exactly 18 distinct classes")
    if len(ids) != PROXY_ROWS or len(set(ids.tolist())) != PROXY_ROWS:
        raise ValueError("Proxy IDs are incomplete or repeated")
    e24_array = np.asarray(e24_scores, dtype=np.float64)
    if e24_array.shape != (PROXY_ROWS, 18) or not np.isfinite(e24_array).all():
        raise ValueError(f"E24 fixed score matrix is invalid: {e24_array.shape}")
    combined = e24_array + row_z(e39_logits, labels, PROXY_ROWS)
    if combined.shape != (PROXY_ROWS, 18) or not np.isfinite(combined).all():
        raise ValueError("Fixed E24 + 1x E39 fusion is invalid")
    predictions = np.asarray(labels, dtype=object)[combined.argmax(axis=1)]
    output = pd.DataFrame({"id": ids, "model": predictions})
    if list(output.columns) != ["id", "model"] or output.isna().any().any():
        raise RuntimeError("Proxy submission schema is invalid")
    if not set(output["model"]).issubset(labels):
        raise RuntimeError("Proxy prediction contains an unknown label")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary proxy output: {temporary}")
    try:
        output.to_parquet(temporary, engine="pyarrow", index=False)
        readback = pd.read_parquet(temporary, engine="pyarrow")
        if not readback.equals(output):
            raise RuntimeError("Proxy Parquet round trip changed IDs or labels")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
