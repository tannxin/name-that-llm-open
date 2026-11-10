"""Audit token lengths on the locked development split without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(np.max(values)),
    }


def audit(root: Path, output_dir: Path, tokenizer_dir: Path) -> None:
    train_path = root / "data/train.csv"
    manifest_path = root / "outputs/e0_tfidf_svm/split_manifest.csv"
    e0_oof_path = root / "outputs/e0_tfidf_svm/oof_predictions.csv"
    roberta_dir = root / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix"
    roberta_oof_path = roberta_dir / "oof_predictions.csv"
    config = json.loads((roberta_dir / "run_config.json").read_text(encoding="utf-8"))

    for name, path in (
        ("train", train_path),
        ("split_manifest", manifest_path),
        ("e0_oof", e0_oof_path),
    ):
        actual = sha256(path)
        expected = config[f"{name}_sha256"]
        if actual != expected:
            raise ValueError(f"{name} SHA-256 differs from the corrected RoBERTa run")

    frame = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    e0_oof = pd.read_csv(e0_oof_path, dtype=str, keep_default_na=False)
    roberta_oof = pd.read_csv(roberta_oof_path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("Unexpected prompt split manifest schema")
    if manifest["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt in split manifest")
    if set(manifest["prompt_id"]) != set(frame["prompt_id"]):
        raise ValueError("Split manifest does not cover train.csv prompts")
    if manifest["split"].value_counts().to_dict() != {"development": 120, "holdout": 30}:
        raise ValueError("The locked development/holdout split changed")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate training ID")

    dev_prompts = set(manifest.loc[manifest["split"].eq("development"), "prompt_id"])
    dev = frame.loc[frame["prompt_id"].isin(dev_prompts), ["id", "prompt_id", "text", "model"]].copy()
    if len(dev) != 2160 or dev["prompt_id"].nunique() != 120:
        raise ValueError("Development split row/prompt count changed")
    if set(dev["id"]) != set(e0_oof["id"]) or set(dev["id"]) != set(roberta_oof["id"]):
        raise ValueError("Development IDs differ from one or both saved OOF files")

    e0_index = e0_oof.set_index("id", verify_integrity=True)
    roberta_index = roberta_oof.set_index("id", verify_integrity=True)
    if not np.array_equal(dev["prompt_id"].to_numpy(), e0_index.loc[dev["id"], "prompt_id"].to_numpy()):
        raise ValueError("E0 OOF prompt IDs differ")
    if not np.array_equal(dev["model"].to_numpy(), e0_index.loc[dev["id"], "true_model"].to_numpy()):
        raise ValueError("E0 OOF labels differ")
    if not np.array_equal(dev["model"].to_numpy(), roberta_index.loc[dev["id"], "true_model"].to_numpy()):
        raise ValueError("RoBERTa OOF labels differ")
    if not np.array_equal(e0_index.loc[dev["id"], "fold"].to_numpy(), roberta_index.loc[dev["id"], "fold"].to_numpy()):
        raise ValueError("Saved E0 and RoBERTa folds differ")
    if config["max_length"] != 256:
        raise ValueError("This audit expects the 256-token corrected RoBERTa run")
    if sha256(tokenizer_dir / "tokenizer.json") != "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e":
        raise ValueError("Tokenizer differs from the pinned RoBERTa training snapshot")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    if tokenizer.name_or_path is None:
        raise ValueError("Could not load the local tokenizer")
    lengths = np.asarray(
        [len(ids) for ids in tokenizer(dev["text"].tolist(), add_special_tokens=True, truncation=False, padding=False)["input_ids"]],
        dtype=np.int32,
    )
    if len(lengths) != len(dev):
        raise ValueError("Tokenizer output length differs")
    dev["token_length"] = lengths
    dev["truncated_256"] = lengths > 256
    dev["truncated_512"] = lengths > 512
    dev["fold"] = e0_index.loc[dev["id"], "fold"].to_numpy()
    dev["roberta_correct"] = (
        roberta_index.loc[dev["id"], "pred_model"].to_numpy() == dev["model"].to_numpy()
    )

    by_class = []
    for label, group in dev.groupby("model", sort=True):
        cut = group["truncated_256"].to_numpy()
        by_class.append({
            "model": label,
            "rows": int(len(group)),
            "truncated_256": int(cut.sum()),
            "truncated_256_fraction": float(cut.mean()),
            "truncated_512": int(group["truncated_512"].sum()),
            "token_length_p50": float(np.quantile(group["token_length"], 0.5)),
            "token_length_p95": float(np.quantile(group["token_length"], 0.95)),
            "token_length_max": int(group["token_length"].max()),
            "roberta_oof_accuracy_all": float(group["roberta_correct"].mean()),
            "roberta_oof_accuracy_truncated_256": (
                float(group.loc[cut, "roberta_correct"].mean()) if cut.any() else None
            ),
            "roberta_oof_accuracy_not_truncated_256": (
                float(group.loc[~cut, "roberta_correct"].mean()) if (~cut).any() else None
            ),
        })

    by_fold = []
    for fold, group in dev.groupby("fold", sort=True):
        by_fold.append({
            "fold": str(fold),
            "rows": int(len(group)),
            "truncated_256": int(group["truncated_256"].sum()),
            "truncated_256_fraction": float(group["truncated_256"].mean()),
        })

    cut = dev["truncated_256"].to_numpy()
    excess = lengths[cut] - 256
    summary = {
        "method": "Pinned original RoBERTa tokenizer, add_special_tokens=True, no truncation; development rows only",
        "source": {
            "train_sha256": sha256(train_path),
            "split_manifest_sha256": sha256(manifest_path),
            "e0_oof_sha256": sha256(e0_oof_path),
            "roberta_oof_sha256": sha256(roberta_oof_path),
            "tokenizer_json_sha256": sha256(tokenizer_dir / "tokenizer.json"),
            "roberta_revision": config["requested_revision"],
        },
        "overall": {
            "rows": int(len(dev)),
            "prompts": int(dev["prompt_id"].nunique()),
            "token_length": distribution(lengths),
            "truncated_256": int(cut.sum()),
            "truncated_256_fraction": float(cut.mean()),
            "truncated_512": int(dev["truncated_512"].sum()),
            "truncated_512_fraction": float(dev["truncated_512"].mean()),
            "truncated_rows_excess_tokens": distribution(excess) if len(excess) else None,
            "roberta_oof_accuracy_truncated_256": float(dev.loc[cut, "roberta_correct"].mean()) if cut.any() else None,
            "roberta_oof_accuracy_not_truncated_256": float(dev.loc[~cut, "roberta_correct"].mean()) if (~cut).any() else None,
        },
        "by_fold": by_fold,
        "by_class": by_class,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(by_class).to_csv(output_dir / "by_class.csv", index=False)
    print(json.dumps(summary["overall"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output_dir, args.tokenizer_dir)
