#!/usr/bin/env python3
"""Audit one small external source without reading official train or holdout rows.

This is a provenance gate, not a training experiment. The current source lacks
Qwen examples and its proxy rows do not identify a serving provider, so it
cannot establish an independently checked 18-class external training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from run_e0 import LABELS


SOURCE_URL = "https://www.kaggle.com/datasets/zeminglii/ntl-own-exact-route-generations"
SOURCE_VERSION = 2
SOURCE_SHA256 = "4ff8c42f3c26b8ee0cfd1552253c3b7faa2ad4d111a8f8781a292e8d41e62afe"
REQUIRED = {
    "text", "model", "prompt_canon", "target_words", "route",
    "provider", "model_seen", "finish_reason", "batch",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def audit(path: Path) -> dict:
    observed_sha256 = sha256(path)
    if observed_sha256 != SOURCE_SHA256:
        raise ValueError("External CSV does not match the audited Kaggle version 2 file")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED - set(frame.columns))}")
    if len(frame) != 2162:
        raise ValueError(f"Expected 2162 rows, got {len(frame)}")
    if not set(frame["model"]).issubset(LABELS):
        raise ValueError("External data contain a noncompetition label")

    label_counts = frame["model"].value_counts().reindex(LABELS, fill_value=0)
    provider_blank = frame["provider"].eq("")
    source_has_prompt_text = "prompt" in frame.columns
    normalized = frame["text"].map(norm_text)
    model_seen_by_label = {
        label: sorted(frame.loc[frame["model"].eq(label), "model_seen"].unique().tolist())
        for label in LABELS if label_counts[label]
    }
    return {
        "source_url": SOURCE_URL,
        "kaggle_version": SOURCE_VERSION,
        "license_on_data_card": "CC BY 4.0",
        "csv_sha256": observed_sha256,
        "rows": len(frame),
        "columns": list(frame.columns),
        "labels_present": int((label_counts > 0).sum()),
        "rows_by_label": {label: int(count) for label, count in label_counts.items()},
        "model_seen_by_label": model_seen_by_label,
        "labels_with_multiple_model_seen_ids": [
            label for label, ids in model_seen_by_label.items() if len(ids) != 1
        ],
        "missing_competition_labels": label_counts.index[label_counts.eq(0)].tolist(),
        "route_provider_counts": [
            {"route": route, "provider": provider, "rows": int(count)}
            for (route, provider), count in frame.groupby(["route", "provider"]).size().items()
        ],
        "model_seen_missing": int(frame["model_seen"].eq("").sum()),
        "provider_missing": int(provider_blank.sum()),
        "prompt_id_missing": int(frame["prompt_canon"].eq("").sum()),
        "prompt_text_in_file": source_has_prompt_text,
        "nonstop_rows": int(frame["finish_reason"].ne("stop").sum()),
        "empty_text_rows": int(normalized.eq("").sum()),
        "exact_duplicate_text_rows": int(normalized.duplicated().sum()),
        "all_18_classes_covered": bool((label_counts > 0).all()),
        "external_training_gate": "no-go",
        "gate_reasons": [
            "Qwen is absent, leaving only 17 of 18 competition labels",
            "1291 Kaggle proxy rows have no serving-provider field",
            "The file has prompt IDs but not prompt text for independent prompt provenance checks",
        ],
        "official_train_or_holdout_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-file", type=Path,
                        default=Path("data/external/zeminglii/passages_v2.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/e12_external_audit/zeminglii_v2.json"))
    args = parser.parse_args()
    report = audit(args.external_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "rows", "labels_present", "missing_competition_labels", "provider_missing",
        "external_training_gate")}, indent=2))


if __name__ == "__main__":
    main()
