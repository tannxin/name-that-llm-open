#!/usr/bin/env python3
"""Pre-registered raw-envelope audit for E29 Qwen 0902 sensitivity branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_prvsiyan_v9 import (
    EXPECTED_HASHES, ROOT, answer_similarity, audit_envelopes,
    prompt_similarity, require, sha256,
)


OUTPUT = Path("outputs/e29_qwen_audit")
MANIFEST = ROOT / "e29_qwen_accepted_manifest.csv"
MODEL = "qwen3.8-max"
SNAPSHOT = "qwen/qwen3.8-max-0902"
PROVIDER = "Alibaba"


def development_texts() -> tuple[list[str], set[str]]:
    split = pd.read_csv("outputs/e0_tfidf_svm/split_manifest.csv", dtype=str,
                        keep_default_na=False)
    dev_prompts = set(split.loc[split["split"].eq("development"), "prompt_id"])
    require(len(dev_prompts) == 120, "Development prompt count changed")
    chunks = pd.read_csv("data/train.csv", usecols=["prompt_id", "text"],
                         dtype=str, keep_default_na=False, chunksize=512)
    dev = pd.concat([part.loc[part["prompt_id"].isin(dev_prompts)]
                     for part in chunks], ignore_index=True)
    require(len(dev) == 2160 and dev["prompt_id"].nunique() == 120,
            "Development response count changed")
    return dev["text"].tolist(), dev_prompts


def main() -> None:
    for name, expected in EXPECTED_HASHES.items():
        require(sha256(ROOT / name) == expected, f"Raw source hash changed: {name}")
    meta = json.loads((ROOT / "kaggle_metadata_v9.json").read_text(encoding="utf-8"))
    require(meta["currentVersionNumber"] == 9 and not meta["isPrivate"],
            "Published source version/visibility changed")
    require("CC BY 4.0" in meta["licenseName"], "Unexpected source license")

    needed = [
        "id", "prompt_id", "model", "text", "prompt_sent", "target_words",
        "split", "provider_route", "requested_model_id", "served_model_id",
        "served_provider", "call_id", "response_id", "finish_reason_norm",
        "source_prompt_id", "is_refusal", "is_truncated", "is_empty",
    ]
    passage_chunks = pd.read_csv(ROOT / "passages.csv", usecols=needed, dtype=str,
                                 keep_default_na=False, chunksize=2000)
    qwen = pd.concat([
        part.loc[part["split"].eq("new") & part["model"].eq(MODEL)]
        for part in passage_chunks
    ], ignore_index=True)
    require(len(qwen) == 300 and qwen["id"].is_unique and qwen["call_id"].is_unique,
            "Expected 300 unique Qwen raw new rows/calls")
    require(qwen["prompt_id"].nunique() == 300 and not qwen["prompt_id"].duplicated().any(),
            "Expected one Qwen row per 300 new prompts")
    require(qwen[["id", "prompt_id", "text", "prompt_sent", "call_id",
                  "served_model_id", "served_provider"]].ne("").all().all(),
            "Qwen rows have blank required source fields")
    require(qwen["provider_route"].eq("openrouter").all()
            and qwen["requested_model_id"].eq(SNAPSHOT).all()
            and qwen["served_model_id"].eq(SNAPSHOT).all()
            and qwen["served_provider"].eq(PROVIDER).all(),
            "Qwen service identity differs from registered snapshot/provider")
    require(qwen["finish_reason_norm"].eq("stop").all()
            and qwen["is_refusal"].eq("False").all()
            and qwen["is_truncated"].eq("False").all()
            and qwen["is_empty"].eq("False").all(),
            "Qwen response stopped abnormally or was flagged")
    norm = qwen["text"].str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)
    require(norm.is_unique, "Qwen raw responses contain exact normalized duplicates")

    prompts = pd.read_csv(ROOT / "prompts.csv", dtype=str, keep_default_na=False)
    prompt_index = prompts.set_index("prompt_id", verify_integrity=True)
    for row in qwen.itertuples(index=False):
        prompt = prompt_index.loc[row.prompt_id]
        require(prompt["split"] == "new" and prompt["prompt_licence"] == "CC BY 4.0",
                f"Unlicensed/non-new prompt {row.id}")
        rendered = prompt["prompt"].replace("{N}", str(int(float(row.target_words))))
        require(rendered == row.prompt_sent and row.source_prompt_id == "",
                f"Prompt rendering/source mismatch {row.id}")

    envelopes = audit_envelopes(qwen)
    dev_texts, dev_prompts = development_texts()
    cosine = answer_similarity(dev_texts, qwen)
    near_duplicate = cosine >= 0.90
    prompt_cosine_max = prompt_similarity(prompts, dev_prompts)
    accepted = qwen.loc[~near_duplicate].copy().sort_values("id").reset_index(drop=True)
    require(len(accepted) > 0, "No eligible Qwen rows remain")

    manifest = accepted[["id", "prompt_id", "model", "call_id", "provider_route",
                         "served_model_id", "served_provider"]].copy()
    manifest["source_file_sha256"] = EXPECTED_HASHES["passages.csv"]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST, index=False)
    report = {
        "audit_passed": True,
        "source_url": "https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/version/9",
        "source_version": 9,
        "source_license": "CC BY 4.0",
        "source_file_sha256": EXPECTED_HASHES,
        "raw_new_qwen_rows": int(len(qwen)),
        "raw_new_qwen_prompts": int(qwen["prompt_id"].nunique()),
        "route": "openrouter",
        "served_model_id": SNAPSHOT,
        "served_provider": PROVIDER,
        "envelopes": envelopes,
        "answer_duplicate_screen": {
            "official_development_rows": len(dev_texts),
            "method": "char-4 TF-IDF, min_df=2, max_features=100000, lowercase=True",
            "maximum_cosine": float(cosine.max()),
            "count_ge_0_5": int((cosine >= 0.5).sum()),
            "count_ge_0_8": int((cosine >= 0.8).sum()),
            "count_ge_0_9": int(near_duplicate.sum()),
        },
        "prompt_overlap_diagnostic": {
            "new_prompts": 300,
            "reconstructed_development_prompts": 120,
            "method": "word 1-2 TF-IDF with English stop words on prompt_body",
            "maximum_cosine": prompt_cosine_max,
        },
        "accepted_rows": int(len(manifest)),
        "accepted_manifest": str(MANIFEST),
        "accepted_manifest_sha256": sha256(MANIFEST),
        "competition_train_derived_source_columns_used": False,
        "locked_holdout_used": False,
        "contest_target_snapshot_proven": False,
        "source_protocol_caveat": "Source prompt length design consulted full official train statistics; contest generation snapshot/provider/settings remain unknown.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps({
        "audit_passed": report["audit_passed"],
        "raw_new_qwen_rows": report["raw_new_qwen_rows"],
        "accepted_rows": report["accepted_rows"],
        "answer_duplicate_screen": report["answer_duplicate_screen"],
        "accepted_manifest_sha256": report["accepted_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
