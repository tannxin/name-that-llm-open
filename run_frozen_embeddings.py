#!/usr/bin/env python3
"""E1: frozen sentence embeddings with the E0 prompt-grouped evaluation split.

The saved holdout prompts are never embedded or scored by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

from run_e0 import LABELS, N_FOLDS, RANDOM_STATE, load_and_validate, sha256


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    repo_id: str
    max_length: int
    prefix: str
    card_max_length: int
    pooling: str = "attention_masked_mean"
    padding_side: str = "right"
    torch_dtype: str | None = None


MODELS = {
    "all-minilm-l6-v2": ModelSpec(
        "all-minilm-l6-v2", "sentence-transformers/all-MiniLM-L6-v2", 256, "", 256
    ),
    # The first comparison uses the same 256-token cap as MiniLM. E5 supports
    # 512 tokens; rerun it at that cap in a separate output directory for an ablation.
    "e5-base-v2": ModelSpec("e5-base-v2", "intfloat/e5-base-v2", 256, "query: ", 512),
    "qwen3-embedding-0.6b": ModelSpec(
        "qwen3-embedding-0.6b",
        "Qwen/Qwen3-Embedding-0.6B",
        256,
        "Instruct: Identify which language model generated the following response.\nQuery:",
        32_768,
        "last_token",
        "left",
        "bfloat16",
    ),
    "qwen3-embedding-4b": ModelSpec(
        "qwen3-embedding-4b",
        "Qwen/Qwen3-Embedding-4B",
        256,
        "Instruct: Identify which language model generated the following response.\nQuery:",
        32_768,
        "last_token",
        "left",
        "bfloat16",
    ),
    # The BGE-M3 repo's 1_Pooling/config.json specifies CLS pooling. The
    # model card says this checkpoint does not need an instruction prefix.
    "bge-m3": ModelSpec(
        "bge-m3", "BAAI/bge-m3", 256, "", 8192, "cls"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e1_frozen"))
    parser.add_argument(
        "--models", nargs="+", choices=MODELS,
        default=["all-minilm-l6-v2", "e5-base-v2"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--qwen-no-instruction", action="store_true",
        help="Ablate Qwen's task instruction; use a separate output directory",
    )
    parser.add_argument(
        "--model-revision",
        help="Pin one selected model to a 40-character Hugging Face commit SHA",
    )
    parser.add_argument(
        "--e5-max-length", type=int, choices=[256, 512], default=256,
        help="Use 256 for the shared first-round cap; 512 requires a separate output directory",
    )
    parser.add_argument(
        "--qwen-max-length", type=int, choices=[256, 512], default=256,
        help="Use 256 for the shared first-round cap; 512 requires a separate output directory",
    )
    parser.add_argument(
        "--bge-max-length", type=int, choices=[256, 512], default=256,
        help="Use 256 for the shared first-round cap; 512 requires a separate output directory",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def load_development(
    train_path: Path, e0_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_and_validate(train_path)
    manifest = pd.read_csv(
        e0_dir / "split_manifest.csv", dtype=str, keep_default_na=False
    )
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("E0 split manifest columns are unexpected")
    if manifest["prompt_id"].duplicated().any():
        raise ValueError("E0 split manifest has repeated prompt IDs")
    if set(manifest["prompt_id"]) != set(frame["prompt_id"]):
        raise ValueError("E0 split manifest does not match the training prompts")
    if not set(manifest["split"]).issubset({"development", "holdout"}):
        raise ValueError("E0 split manifest contains an unknown split")
    if manifest["split"].eq("holdout").sum() != 30:
        raise ValueError("Expected exactly 30 untouched holdout prompts")
    development_prompts = set(
        manifest.loc[manifest["split"].eq("development"), "prompt_id"]
    )
    development = frame.loc[
        frame["prompt_id"].isin(development_prompts)
    ].reset_index(drop=True)
    if development["prompt_id"].nunique() != 120:
        raise ValueError("Expected exactly 120 development prompts")

    e0_oof = pd.read_csv(
        e0_dir / "oof_predictions.csv", dtype=str, keep_default_na=False
    )
    if not e0_oof["id"].is_unique or len(e0_oof) != len(development):
        raise ValueError("E0 OOF rows are missing or duplicated")
    if e0_oof["id"].tolist() != development["id"].tolist():
        raise ValueError("E0 OOF order or row IDs differ from development data")
    if e0_oof["prompt_id"].tolist() != development["prompt_id"].tolist():
        raise ValueError("E0 OOF prompt IDs differ from development data")
    if e0_oof["true_model"].tolist() != development["model"].tolist():
        raise ValueError("E0 OOF labels differ from development data")
    return frame, development, e0_oof


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but this PyTorch installation cannot use it")
    return requested


def get_revision(repo_id: str) -> str:
    revision = HfApi().model_info(repo_id).sha
    if not revision or len(revision) != 40:
        raise RuntimeError(f"Could not resolve a commit SHA for {repo_id}")
    return revision


def encode_development(
    texts: list[str], spec: ModelSpec, revision: str, device: str, batch_size: int
) -> tuple[np.ndarray, dict[str, int | float | str | None]]:
    cuda_index = torch.device(device).index if device.startswith("cuda") else None
    if device.startswith("cuda"):
        # PyTorch 2.6 on the server requires an initialized CUDA allocator
        # before reset_peak_memory_stats accepts the selected device.
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id, revision=revision, padding_side=spec.padding_side
    )
    model_kwargs = {}
    if spec.torch_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModel.from_pretrained(spec.repo_id, revision=revision, **model_kwargs)
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize(cuda_index)
    model_load_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    chunks: list[np.ndarray] = []
    truncated_rows = 0
    max_tokens_before_truncation = 0
    prefix_tokens = len(
        tokenizer(spec.prefix, add_special_tokens=False)["input_ids"]
    ) if spec.prefix else 0

    for start in range(0, len(texts), batch_size):
        batch_texts = [spec.prefix + item for item in texts[start : start + batch_size]]
        # Count tokens before truncation to record how much text each model sees.
        lengths = [
            len(tokens)
            for tokens in tokenizer(
                batch_texts, add_special_tokens=True, truncation=False
            )["input_ids"]
        ]
        truncated_rows += sum(length > spec.max_length for length in lengths)
        max_tokens_before_truncation = max(max_tokens_before_truncation, *lengths)
        encoded = tokenizer(
            batch_texts,
            max_length=spec.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
            if spec.pooling == "last_token":
                if not bool(torch.all(encoded["attention_mask"][:, -1] == 1).item()):
                    raise ValueError("Left-padded input has a padded final token")
                pooled = output[:, -1]
            elif spec.pooling == "attention_masked_mean":
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif spec.pooling == "cls":
                pooled = output[:, 0]
            else:
                raise ValueError(f"Unknown pooling method: {spec.pooling}")
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        chunks.append(normalized.float().cpu().numpy())
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(texts):
            print(
                f"{spec.slug}: embedded {min(start + batch_size, len(texts))}/{len(texts)}",
                flush=True,
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize(cuda_index)
        peak_allocated = torch.cuda.max_memory_allocated(cuda_index)
        peak_reserved = torch.cuda.max_memory_reserved(cuda_index)
    else:
        peak_allocated = None
        peak_reserved = None
    inference_seconds = time.perf_counter() - inference_start
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0), {
        "truncated_rows": truncated_rows,
        "max_tokens_before_truncation": max_tokens_before_truncation,
        "input_prefix_tokens": prefix_tokens,
        "model_load_seconds": round(model_load_seconds, 3),
        "embedding_inference_seconds": round(inference_seconds, 3),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "attention_implementation": attention_implementation,
    }


def get_embeddings(
    development: pd.DataFrame,
    spec: ModelSpec,
    revision: str,
    data_sha: str,
    model_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int | float | str | None]]:
    cache_file = model_dir / "development_embeddings.npz"
    metadata_file = model_dir / "embedding_cache.json"
    expected = {
        "model_id": spec.repo_id,
        "model_revision": revision,
        "max_length": spec.max_length,
        "prefix": spec.prefix,
        "pooling": spec.pooling,
        "normalization": "L2",
        "train_sha256": data_sha,
        "development_ids_sha256": hashlib.sha256(
            "\n".join(development["id"]).encode("utf-8")
        ).hexdigest(),
    }
    if spec.pooling == "last_token":
        expected["padding_side"] = spec.padding_side
        expected["torch_dtype"] = spec.torch_dtype
    if cache_file.exists() and metadata_file.exists():
        cache_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if all(cache_metadata.get(key) == value for key, value in expected.items()):
            with np.load(cache_file) as contents:
                embeddings = contents["embeddings"]
            if embeddings.shape[0] != len(development) or not np.isfinite(embeddings).all():
                raise ValueError("Cached embeddings are incomplete or non-finite")
            print(f"{spec.slug}: reused {cache_file}", flush=True)
            stats = cache_metadata["token_length_stats"].copy()
            if "longest_untruncated_tokens" in stats:
                stats["max_tokens_before_truncation"] = stats.pop(
                    "longest_untruncated_tokens"
                )
            return embeddings, stats
        raise ValueError(f"Existing embedding cache has different inputs: {cache_file}")

    embeddings, stats = encode_development(
        development["text"].tolist(), spec, revision, device, batch_size
    )
    if embeddings.shape[0] != len(development) or not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding computation was incomplete or produced non-finite values")
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, embeddings=embeddings)
    os.replace(temporary, cache_file)
    metadata_file.write_text(
        json.dumps({**expected, "token_length_stats": stats}, indent=2) + "\n",
        encoding="utf-8",
    )
    return embeddings, stats


def make_classifier(seed: int) -> LinearSVC:
    return LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=True,
        max_iter=10_000,
        random_state=seed,
    )


def verify_weight_files(spec: ModelSpec, revision: str) -> list[dict[str, str]]:
    """Hash every model weight blob used by the pinned Hub snapshot."""
    snapshot = (
        Path(HF_HUB_CACHE)
        / ("models--" + spec.repo_id.replace("/", "--"))
        / "snapshots"
        / revision
    )
    index_file = snapshot / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        filenames = sorted(set(index["weight_map"].values()))
        if not filenames:
            raise ValueError(f"No weight shards in {index_file}")
    else:
        filenames = [
            filename for filename in ("model.safetensors", "pytorch_model.bin")
            if (snapshot / filename).is_file()
        ]
        if len(filenames) != 1:
            raise ValueError(f"Expected one unsharded model weight in {snapshot}")
    records = []
    for filename in filenames:
        weight_file = snapshot / filename
        if not weight_file.is_file():
            raise ValueError(f"Missing model weight shard: {weight_file}")
        digest = sha256(weight_file)
        blob_name = weight_file.resolve().name
        if len(blob_name) == 64 and all(
            char in "0123456789abcdef" for char in blob_name
        ) and digest != blob_name:
            raise ValueError(f"Hub weight blob SHA-256 mismatch: {weight_file}")
        records.append({"filename": filename, "sha256": digest})
    return records


def evaluate(
    frame: pd.DataFrame,
    development: pd.DataFrame,
    e0_oof: pd.DataFrame,
    embeddings: np.ndarray,
    spec: ModelSpec,
    revision: str,
    stats: dict[str, int | float | str | None],
    train_path: Path,
    train_sha: str,
    model_dir: Path,
    e0_dir: Path,
    device: str,
    seed: int,
) -> None:
    labels = development["model"].to_numpy()
    groups = development["prompt_id"].to_numpy()
    predictions = np.empty(len(development), dtype=object)
    decisions = np.empty((len(development), len(LABELS)), dtype=np.float32)
    fold_for_row = np.zeros(len(development), dtype=np.int8)
    fold_rows = []
    for fold, (fit_idx, valid_idx) in enumerate(
        GroupKFold(n_splits=N_FOLDS).split(embeddings, labels, groups), start=1
    ):
        fit_groups = set(groups[fit_idx])
        valid_groups = set(groups[valid_idx])
        if fit_groups & valid_groups:
            raise RuntimeError(f"Fold {fold} has overlapping prompt groups")
        if set(labels[fit_idx]) != set(LABELS) or set(labels[valid_idx]) != set(LABELS):
            raise ValueError(f"Fold {fold} is missing at least one official class")
        if not np.all(e0_oof.iloc[valid_idx]["fold"].eq(str(fold))):
            raise ValueError(f"Fold {fold} differs from saved E0 assignments")
        head = make_classifier(seed)
        head.fit(embeddings[fit_idx], labels[fit_idx])
        predictions[valid_idx] = head.predict(embeddings[valid_idx])
        class_positions = [np.where(head.classes_ == label)[0][0] for label in LABELS]
        decisions[valid_idx] = head.decision_function(embeddings[valid_idx])[
            :, class_positions
        ]
        fold_for_row[valid_idx] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(fit_idx)),
                "validation_rows": int(len(valid_idx)),
                "train_prompts": len(fit_groups),
                "validation_prompts": len(valid_groups),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[valid_idx], predictions[valid_idx])
                ),
            }
        )
        print(
            f"{spec.slug} fold {fold}/{N_FOLDS}: "
            f"balanced_accuracy={fold_rows[-1]['balanced_accuracy']:.6f}",
            flush=True,
        )
    if not np.all(fold_for_row > 0):
        raise RuntimeError("OOF predictions did not cover every development row")
    confusion = confusion_matrix(labels, predictions, labels=LABELS)
    class_metrics = pd.DataFrame(
        {
            "model": LABELS,
            "support": confusion.sum(axis=1),
            "correct": confusion.diagonal(),
        }
    )
    class_metrics["recall"] = class_metrics["correct"] / class_metrics["support"]
    oof = development.loc[:, ["id", "prompt_id"]].copy()
    oof["true_model"] = labels
    oof["pred_model"] = predictions
    oof["fold"] = fold_for_row
    oof["correct"] = labels == predictions
    oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    score_frame = oof.loc[:, ["id", "prompt_id", "true_model", "fold"]].copy()
    for column, label in enumerate(LABELS):
        score_frame[f"score_{label}"] = decisions[:, column]
    score_frame.to_csv(model_dir / "decision_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(model_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(confusion, index=LABELS, columns=LABELS).rename_axis(
        index="true_model", columns="pred_model"
    ).to_csv(model_dir / "confusion_matrix.csv")
    class_metrics.to_csv(model_dir / "class_metrics.csv", index=False)
    development_head = make_classifier(seed)
    development_head.fit(embeddings, labels)
    joblib.dump(development_head, model_dir / "dev_head.joblib", compress=3)
    weight_files = verify_weight_files(spec, revision)
    metrics = {
        "experiment": "e1_frozen",
        "model_slug": spec.slug,
        "model_id": spec.repo_id,
        "model_revision": revision,
        "model_weight_file": weight_files[0]["filename"] if len(weight_files) == 1 else None,
        "model_weight_sha256": weight_files[0]["sha256"] if len(weight_files) == 1 else None,
        "model_weight_files": weight_files,
        "huggingface_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "metric": "balanced_accuracy",
        "oof_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mean_fold_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in fold_rows])
        ),
        "fold_metrics": fold_rows,
        "n_rows_total": len(frame),
        "n_rows_development": len(development),
        "n_prompts_development": int(development["prompt_id"].nunique()),
        "n_prompts_holdout": 30,
        "n_folds": N_FOLDS,
        "n_classes": len(LABELS),
        "labels": list(LABELS),
        "train_file": str(train_path.resolve()),
        "train_sha256": train_sha,
        "e0_split_manifest": str((e0_dir / "split_manifest.csv").resolve()),
        "encoder_input_prefix": spec.prefix,
        "encoder_max_length": spec.max_length,
        "encoder_card_max_length": spec.card_max_length,
        "encoder_pooling": spec.pooling,
        "encoder_padding_side": spec.padding_side,
        "encoder_model_dtype": spec.torch_dtype,
        "encoder_normalization": "L2",
        "token_length_stats": stats,
        "classifier": "LinearSVC(C=1.0, class_weight='balanced', dual=True, max_iter=10000)",
        "seed": seed,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{spec.slug} OOF balanced_accuracy={metrics['oof_balanced_accuracy']:.6f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.model_revision and (
        len(args.models) != 1
        or len(args.model_revision) != 40
        or not all(char in "0123456789abcdef" for char in args.model_revision)
    ):
        raise ValueError("--model-revision requires one model and a 40-character lowercase SHA")
    if args.qwen_no_instruction and (
        len(args.models) != 1 or not args.models[0].startswith("qwen3-embedding-")
    ):
        raise ValueError("--qwen-no-instruction requires one Qwen model")
    frame, development, e0_oof = load_development(args.train, args.e0_dir)
    train_sha = sha256(args.train)
    device = resolve_device(args.device)
    print(f"Development: {len(development)} rows; device={device}", flush=True)
    for slug in args.models:
        spec = MODELS[slug]
        if args.qwen_no_instruction:
            spec = replace(spec, prefix="")
        if slug == "e5-base-v2" and args.e5_max_length != spec.max_length:
            spec = replace(spec, max_length=args.e5_max_length)
        if slug.startswith("qwen3-embedding-") and args.qwen_max_length != spec.max_length:
            spec = replace(spec, max_length=args.qwen_max_length)
        if slug == "bge-m3" and args.bge_max_length != spec.max_length:
            spec = replace(spec, max_length=args.bge_max_length)
        revision = args.model_revision or get_revision(spec.repo_id)
        model_dir = args.output_dir / spec.slug
        embeddings, stats = get_embeddings(
            development,
            spec,
            revision,
            train_sha,
            model_dir,
            device,
            args.batch_size,
        )
        evaluate(
            frame,
            development,
            e0_oof,
            embeddings,
            spec,
            revision,
            stats,
            args.train,
            train_sha,
            model_dir,
            args.e0_dir,
            device,
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
