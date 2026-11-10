#!/usr/bin/env python3
"""Read-only E26 source-shift audit for the fixed E24 external sample.

Only the 120-prompt development portion of official train is inspected. The
three-row Kaggle preview test is deliberately excluded from inference about
hidden-test distribution. No external funnel or competition-derived columns
are read from the source CSV.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN = Path("data/train.csv")
SOURCE = Path("data/external/prvsiyan/passages.csv")
SELECTED = Path("outputs/e24_prvsiyan_fit/selected_external.csv")
SPLIT = Path("outputs/e0_tfidf_svm/split_manifest.csv")
OUTPUT = Path("outputs/e26_domain_audit")
EXPECTED_TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_SOURCE_SHA = "393f300c7b521507a55f5a8793bd2256519ed6324da97153f8cd2b3503f01ce0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    text = result["text"].fillna("").astype(str)
    result["words"] = text.map(lambda s: len(s.split()))
    result["characters"] = text.str.len()
    result["chars_per_word"] = result["characters"] / result["words"].clip(lower=1)
    result["newlines"] = text.str.contains("\n", regex=False)
    result["em_dash"] = text.str.contains("—", regex=False)
    result["curly_apostrophe"] = text.str.contains("’", regex=False)
    result["semicolon"] = text.str.contains(";", regex=False)
    result["colon"] = text.str.contains(":", regex=False)
    result["markdown_marker"] = text.map(
        lambda s: bool(re.search(r"\*\*|`|(?:^|\n)\s*(?:#|[-*])\s", s))
    )
    return result


def metric_summary(frame: pd.DataFrame) -> dict:
    result = {"n_rows": int(len(frame)), "n_prompts": int(frame["prompt_id"].nunique())}
    for name in ("words", "characters", "chars_per_word"):
        values = frame[name].to_numpy(dtype=float)
        result[name] = {
            "p10": round(float(np.quantile(values, 0.10)), 2),
            "median": round(float(np.median(values)), 2),
            "p90": round(float(np.quantile(values, 0.90)), 2),
            "mean": round(float(np.mean(values)), 2),
        }
    for name in ("newlines", "em_dash", "curly_apostrophe", "semicolon", "colon", "markdown_marker"):
        result[name + "_rate"] = round(float(frame[name].mean()), 4)
    edges = [-1, 39, 79, 119, 159, 199, 249, float("inf")]
    names = ["<40", "40-79", "80-119", "120-159", "160-199", "200-249", "250+"]
    groups = pd.cut(frame["words"], bins=edges, labels=names)
    result["word_bins"] = {name: round(float((groups == name).mean()), 4) for name in names}
    return result


def main() -> None:
    require(sha256(TRAIN) == EXPECTED_TRAIN_SHA, "Official train SHA changed")
    require(sha256(SOURCE) == EXPECTED_SOURCE_SHA, "E24 raw external source SHA changed")
    split = pd.read_csv(SPLIT, dtype=str, keep_default_na=False)
    dev_ids = set(split.loc[split["split"].eq("development"), "prompt_id"])
    require(len(dev_ids) == 120, "Development prompt set changed")

    official = pd.concat(
        [chunk.loc[chunk["prompt_id"].isin(dev_ids)]
         for chunk in pd.read_csv(
             TRAIN, usecols=["prompt_id", "text", "model"],
             dtype=str, keep_default_na=False, chunksize=512,
         )],
        ignore_index=True,
    )
    require(len(official) == 2160, "Development rows changed")
    require(official["prompt_id"].nunique() == 120, "Development prompts changed")

    selected = pd.read_csv(SELECTED, dtype=str, keep_default_na=False)
    require(len(selected) == 3620 and selected["id"].is_unique, "E24 selected sample changed")
    source = pd.read_csv(
        SOURCE,
        usecols=["id", "prompt_id", "model", "text", "split", "domain", "task_type", "style", "target_words", "provider_route", "served_model_id"],
        dtype=str, keep_default_na=False,
    )
    source = source.loc[source["id"].isin(set(selected["id"]))].copy()
    require(len(source) == 3620 and source["id"].is_unique, "Raw E24 sample IDs changed")
    joined = source.merge(selected[["id", "prompt_id", "model", "served_model_id", "provider_route"]],
                          on="id", how="inner", validate="one_to_one", suffixes=("_raw", "_selected"))
    for field in ("prompt_id", "model", "served_model_id", "provider_route"):
        require((joined[field + "_raw"] == joined[field + "_selected"]).all(),
                f"E24 selected source mismatch: {field}")
    require(source["split"].eq("new").all(), "Sample includes non-new split")
    require(source["model"].nunique() == 17, "E24 class coverage changed")
    require(not source["model"].eq("qwen3.8-max").any(), "Qwen unexpectedly included")
    require(source["prompt_id"].nunique() == 250, "E24 prompt sample changed")

    official = add_features(official)
    source = add_features(source)
    class_rows = []
    for label in sorted(official["model"].unique()):
        a = official.loc[official["model"].eq(label)]
        b = source.loc[source["model"].eq(label)]
        class_rows.append({
            "model": label,
            "official_n": len(a), "external_n": len(b),
            "official_words_median": float(a["words"].median()),
            "external_words_median": float(b["words"].median()) if len(b) else None,
            "median_word_shift": float(b["words"].median() - a["words"].median()) if len(b) else None,
            "official_words_p90": float(a["words"].quantile(0.9)),
            "external_words_p90": float(b["words"].quantile(0.9)) if len(b) else None,
            "official_em_dash_rate": float(a["em_dash"].mean()),
            "external_em_dash_rate": float(b["em_dash"].mean()) if len(b) else None,
            "official_curly_apostrophe_rate": float(a["curly_apostrophe"].mean()),
            "external_curly_apostrophe_rate": float(b["curly_apostrophe"].mean()) if len(b) else None,
        })
    classes = pd.DataFrame(class_rows)
    official_covered = official.loc[official["model"].isin(set(source["model"]))]
    domains = source[["prompt_id", "domain", "task_type", "style", "target_words"]].drop_duplicates("prompt_id")
    report = {
        "purpose": "source shift diagnostic, not a new validation score",
        "official_train_sha256": EXPECTED_TRAIN_SHA,
        "external_source_sha256": EXPECTED_SOURCE_SHA,
        "selected_external_sha256": sha256(SELECTED),
        "official_dev_18_classes": metric_summary(official),
        "official_dev_17_covered_classes": metric_summary(official_covered),
        "external_selected_17_classes": metric_summary(source),
        "external_prompt_domains": domains["domain"].value_counts().to_dict(),
        "external_prompt_task_types": domains["task_type"].value_counts().to_dict(),
        "external_prompt_styles": domains["style"].value_counts().head(20).to_dict(),
        "external_target_words": pd.to_numeric(domains["target_words"], errors="coerce").describe().round(2).to_dict(),
        "locked_holdout_inspected": False,
        "hidden_test_inspected": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    classes.to_csv(OUTPUT / "class_shift.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nLargest median word-count shifts:")
    print(classes.dropna(subset=["median_word_shift"])
          .assign(abs_shift=lambda x: x["median_word_shift"].abs())
          .nlargest(8, "abs_shift")
          [["model", "official_words_median", "external_words_median", "median_word_shift"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
