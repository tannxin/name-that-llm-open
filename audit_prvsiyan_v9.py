#!/usr/bin/env python3
"""Audit the original prvsiyan v9 `new` split before external-data fitting.

This reads only unlabeled official development texts for the near-duplicate
screen. It never uses the source's competition-train-derived funnel columns.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from run_e0 import LABELS


ROOT = Path("data/external/prvsiyan")
OUTPUT = Path("outputs/e24_prvsiyan_audit")
EXPECTED_HASHES = {
    "passages.csv": "393f300c7b521507a55f5a8793bd2256519ed6324da97153f8cd2b3503f01ce0",
    "prompts.csv": "d1424468ca172e3e589a996c8a2f9a0ee5339dc3443104ba05e27bd1cde3be48",
    "provenance_summary.json": "53243ba071141186be15fb91fe92cba9eb305ac89b9824bedc9cc97dc67a0865",
    "provenance.jsonl": "a7e78ff29ba2510fc9180cfcf864e0a1ecc1c36ee1bbd059785b2ad95b2a94de",
}
NEAR_DUPLICATE_THRESHOLD = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_route(row: object) -> bool:
    model = row.model
    if model.startswith("claude-"):
        return (row.provider_route == "anthropic"
                and row.served_provider == "Anthropic (first-party API)"
                and row.served_model_id == model)
    if model.startswith("gemini-"):
        return (row.provider_route == "google"
                and row.served_provider.startswith("Google Vertex AI")
                and row.served_model_id == model)
    if model.startswith("gpt-"):
        served = "gpt-5.5-2026-04-23" if model == "gpt-5.5" else model
        return (row.provider_route == "openai"
                and row.served_provider == "OpenAI (first-party API)"
                and row.served_model_id == served)
    if model.startswith("glm-"):
        return (row.provider_route == "openrouter"
                and row.served_provider == "Z.AI"
                and row.served_model_id == "z-ai/" + model)
    if model == "kimi-k3":
        return (row.provider_route == "openrouter"
                and row.served_provider == "Moonshot AI"
                and row.served_model_id == "moonshotai/kimi-k3")
    return False  # Qwen's -0902 snapshot has not been mapped to the contest target.


def load_development_texts() -> tuple[list[str], set[str]]:
    split = pd.read_csv("outputs/e0_tfidf_svm/split_manifest.csv", dtype=str,
                        keep_default_na=False)
    dev_prompts = set(split.loc[split["split"].eq("development"), "prompt_id"])
    require(len(dev_prompts) == 120, "Development prompt manifest changed")
    official = pd.read_csv("data/train.csv", usecols=["prompt_id", "text"],
                           dtype=str, keep_default_na=False)
    dev = official.loc[official["prompt_id"].isin(dev_prompts), "text"]
    require(len(dev) == 2160, "Development response count changed")
    return dev.tolist(), dev_prompts


def answer_similarity(dev_texts: list[str], external: pd.DataFrame) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(4, 4), min_df=2,
        max_features=100_000, lowercase=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(dev_texts + external["text"].tolist())
    n_dev = len(dev_texts)
    maxima = np.zeros(len(external), dtype=np.float32)
    for start in range(0, len(external), 128):
        stop = min(start + 128, len(external))
        maxima[start:stop] = (matrix[n_dev + start:n_dev + stop]
                              @ matrix[:n_dev].T).toarray().max(axis=1)
    return maxima


def prompt_similarity(prompts: pd.DataFrame, dev_prompts: set[str]) -> float:
    new = prompts.loc[prompts["split"].eq("new"), "prompt_body"].tolist()
    reconstructed_ids = {"tr_" + value for value in dev_prompts}
    reconstructed = prompts.loc[prompts["prompt_id"].isin(reconstructed_ids),
                                "prompt_body"].tolist()
    require(len(new) == 300 and len(reconstructed) == 120,
            "Prompt banks changed")
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), stop_words="english",
        lowercase=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(new + reconstructed)
    return float((matrix[:300] @ matrix[300:].T).toarray().max())


def audit_envelopes(external: pd.DataFrame) -> dict:
    needed = set(external["call_id"])
    matched = {}
    total = 0
    with (ROOT / "provenance.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            envelope = json.loads(line)
            total += 1
            call_id = envelope.get("call_id")
            if call_id in needed:
                require(call_id not in matched, f"Repeated raw call ID {call_id}")
                matched[call_id] = envelope
    require(total == 13213, "Unexpected raw envelope count")
    require(len(matched) == len(external), "Missing raw envelope for a new row")
    require(external["call_id"].is_unique, "Nonunique call ID in passages CSV")

    for row in external.itertuples(index=False):
        raw = matched[row.call_id]
        for column, raw_field in (
            ("prompt_id", "prompt_id"), ("model", "label"),
            ("provider_route", "route"),
            ("requested_model_id", "requested_model_id"),
            ("served_model_id", "served_model_id"),
            ("served_provider", "served_provider"),
            ("text", "text"), ("response_id", "response_id"),
            ("finish_reason_norm", "finish_reason_norm"),
            ("split", "published_split"),
        ):
            require(getattr(row, column) == raw.get(raw_field),
                    f"Raw/CSV mismatch in {column} for {row.id}")
        require(raw.get("status") == "ok"
                and raw.get("identity", {}).get("ok") is True
                and raw.get("response", {}).get("http_status") == 200,
                f"Unverified raw call for {row.id}")
        body = raw["request"]["body"]
        require(body.get("model") == row.requested_model_id,
                f"Request model mismatch for {row.id}")
        prompt = (body.get("contents") if row.provider_route == "google"
                  else body["messages"][0]["content"])
        require(prompt == row.prompt_sent, f"Request prompt mismatch for {row.id}")
        response = raw["response"]["json"]
        actual_model = (response.get("model_version")
                        if row.provider_route == "google" else response.get("model"))
        require(actual_model == row.served_model_id,
                f"Actual response model mismatch for {row.id}")
        if row.provider_route == "openrouter":
            require(response.get("provider") == row.served_provider,
                    f"Actual response provider mismatch for {row.id}")
            pin = body.get("provider", {})
            require(pin.get("allow_fallbacks") is False
                    and len(pin.get("order", [])) == 1
                    and pin.get("order") == pin.get("only"),
                    f"OpenRouter provider not pinned for {row.id}")
    return {"all_envelopes": total, "new_calls_matched": len(matched),
            "mismatched_fields": 0, "missing_calls": 0,
            "response_model_field": "model_version for Vertex; model otherwise"}


def main() -> None:
    for name, expected in EXPECTED_HASHES.items():
        require(sha256(ROOT / name) == expected, f"Source file hash changed: {name}")
    meta = json.loads((ROOT / "kaggle_metadata_v9.json").read_text(encoding="utf-8"))
    require(meta["currentVersionNumber"] == 9 and not meta["isPrivate"],
            "Kaggle version or access differs from audit")
    require("CC BY 4.0" in meta["licenseName"], "Unexpected source license")

    passages = pd.read_csv(ROOT / "passages.csv", dtype=str, keep_default_na=False)
    prompts = pd.read_csv(ROOT / "prompts.csv", dtype=str, keep_default_na=False)
    require(len(passages) == 13212 and len(prompts) == 774,
            "Original source row counts changed")
    external = passages.loc[passages["split"].eq("new")].copy().reset_index(drop=True)
    require(len(external) == 5400 and external["id"].is_unique,
            "Expected 5400 unique raw new rows")
    require(external["prompt_id"].nunique() == 300, "Expected 300 new prompts")
    counts = external["model"].value_counts().reindex(LABELS, fill_value=0)
    require(counts.eq(300).all(), "Expected 300 raw passages per contest class")
    require(external[["id", "prompt_id", "text", "prompt_sent", "call_id",
                      "served_model_id", "served_provider"]].ne("").all().all(),
            "Blank required source field")
    require(external["finish_reason_norm"].eq("stop").all(),
            "Non-stop source passage")
    require(external["is_refusal"].eq("False").all()
            and external["is_truncated"].eq("False").all()
            and external["is_empty"].eq("False").all(),
            "Source contains marked refusal, truncation or empty passage")
    normalized = external["text"].str.strip().str.casefold().str.replace(
        r"\s+", " ", regex=True,
    )
    require(normalized.is_unique, "Duplicate normalized external passage")

    prompt_index = prompts.set_index("prompt_id", verify_integrity=True)
    for row in external.itertuples(index=False):
        prompt = prompt_index.loc[row.prompt_id]
        require(prompt["split"] == "new" and prompt["prompt_licence"] == "CC BY 4.0",
                f"Unlicensed or non-new source prompt for {row.id}")
        rendered = prompt["prompt"].replace("{N}", str(int(float(row.target_words))))
        require(rendered == row.prompt_sent, f"Prompt bank mismatch for {row.id}")
        require(row.source_prompt_id == "", f"Competition prompt link for {row.id}")

    envelope_audit = audit_envelopes(external)
    dev_texts, dev_prompts = load_development_texts()
    cosine = answer_similarity(dev_texts, external)
    prompt_cosine_max = prompt_similarity(prompts, dev_prompts)
    eligible_route = external.apply(expected_route, axis=1).to_numpy(dtype=bool)
    near_duplicate = cosine >= NEAR_DUPLICATE_THRESHOLD
    accepted = external.loc[eligible_route & ~near_duplicate].copy()
    accepted_counts = accepted["model"].value_counts().reindex(LABELS, fill_value=0)
    require(len(accepted) == 4322, "Eligible source count changed")
    require(accepted_counts.eq(0).sum() == 1
            and accepted_counts["qwen3.8-max"] == 0,
            "Only the Qwen snapshot should be absent")

    manifest = accepted[["id", "prompt_id", "model", "call_id",
                         "provider_route", "served_model_id", "served_provider"]].copy()
    manifest = manifest.sort_values("id").reset_index(drop=True)
    manifest["source_file_sha256"] = EXPECTED_HASHES["passages.csv"]
    target = ROOT / "e24_accepted_manifest.csv"
    manifest.to_csv(target, index=False)
    report = {
        "audit_passed": True,
        "source_url": "https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset",
        "source_version": 9,
        "license": "CC BY 4.0",
        "source_file_sha256": EXPECTED_HASHES,
        "raw_new_rows": len(external),
        "raw_new_prompts": external["prompt_id"].nunique(),
        "raw_new_rows_per_class": 300,
        "envelopes": envelope_audit,
        "answer_duplicate_screen": {
            "official_development_rows": len(dev_texts),
            "method": "char-4 TF-IDF, min_df=2, max_features=100000, lowercase=True",
            "maximum_cosine": float(cosine.max()),
            "count_ge_0_5": int((cosine >= 0.5).sum()),
            "count_ge_0_8": int((cosine >= 0.8).sum()),
            "count_ge_0_9": int(near_duplicate.sum()),
        },
        "prompt_overlap_diagnostic": {
            "new_prompts": 300, "reconstructed_development_prompts": 120,
            "method": "word 1-2 TF-IDF with English stop words on prompt_body",
            "maximum_cosine": prompt_cosine_max,
        },
        "route_rule": "First-party Anthropic/OpenAI/Google Vertex, or OpenRouter pinned Z.AI/Moonshot; exclude Kaggle proxy, OpenRouter Gemini and Qwen -0902",
        "excluded_kaggle_proxy": int(external["provider_route"].eq("kaggle_proxy").sum()),
        "excluded_qwen_snapshot": int(external["model"].eq("qwen3.8-max").sum()),
        "excluded_openrouter_gemini": int((external["model"].str.startswith("gemini-")
                                             & external["provider_route"].eq("openrouter")).sum()),
        "excluded_near_duplicate": int(near_duplicate.sum()),
        "accepted_rows": len(manifest),
        "accepted_by_class": {label: int(count) for label, count in accepted_counts.items()},
        "accepted_manifest": str(target),
        "accepted_manifest_sha256": sha256(target),
        "competition_train_derived_source_columns_used": False,
        "source_protocol_caveat": "The source designed target lengths using the full official train distribution; OOF is exploratory, not independent of all official train information.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "audit_passed", "raw_new_rows", "accepted_rows", "accepted_by_class",
        "accepted_manifest_sha256",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
