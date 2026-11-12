#!/usr/bin/env python3
"""Unscored E39 fold-1 adapter load and 4000-text Kaggle T4 deployment probe.

`reference` runs once on the training server against three fold-1 TRAIN-side
texts. `environment`, `smoke`, and `full` run in an Internet-off Kaggle GPU
Notebook. This file never reads a target label or computes accuracy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e39_probe_common as common  # noqa: E402


MAX_LENGTH = 256
SMOKE_ATOL = 0.05  # Fixed before any Kaggle T4 run; argmax must also agree.
SMOKE_TRAIN_ROW_INDICES = (1, 3, 4)
E0_OOF_SHA256 = "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa"
OFFICIAL_LABELS = (
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6",
    "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max",
    "glm-5.3", "glm-5.3-flash", "kimi-k3",
)
MIN_T4_MARGIN_BYTES = 256 * 2**20
MAX_FULL_INFERENCE_SECONDS = 3 * 3600
RUNTIME = {
    "torch": "2.6.0", "transformers": "4.51.3", "peft": "0.15.2",
    "accelerate": "1.6.0", "bitsandbytes": "0.45.5",
}


def read_manifest(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("format_version") != 1 or document.get("fold") != 1
            or document.get("model_repo") != "Qwen/Qwen3-8B-Base"
            or document.get("model_revision") != "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
            or document.get("train_sha256") != common.TRAIN_SHA256
            or document.get("labels") != list(OFFICIAL_LABELS)
            or len(document.get("model_files", {})) != 12
            or "adapter_model.safetensors" not in document.get("adapter_files", {})
            or document.get("purpose") != "unscored_kaggle_t4_deployment_probe_only"):
        raise ValueError("E39 deployment asset manifest differs from locked fold 1")
    return document


def verify_assets(model_dir: Path, adapter_dir: Path, manifest: dict) -> None:
    for directory, entries in ((model_dir, manifest["model_files"]),
                               (adapter_dir, manifest["adapter_files"])):
        root = directory.resolve(strict=True)
        for name, value in entries.items():
            relative = Path(name)
            if relative.name != name or name.startswith(".") or ".." in relative.parts:
                raise ValueError(f"Unsafe asset name: {name}")
            path = directory / name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Missing or redirected E39 asset: {name}")
            expected = value["sha256"] if isinstance(value, dict) else value
            if isinstance(value, dict) and path.stat().st_size != value["size"]:
                raise ValueError(f"E39 asset size changed: {name}")
            if common.sha256_file(path) != expected:
                raise ValueError(f"E39 asset SHA-256 changed: {name}")
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("r") != 16 or config.get("lora_alpha") != 32
            or config.get("task_type") != "SEQ_CLS"):
        raise ValueError("E39 fold adapter configuration differs from locked training")
    base_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if base_config.get("model_type") != "qwen3" or base_config.get("num_hidden_layers") != 36:
        raise ValueError("E39 pinned Base architecture differs from 36-layer Qwen3")


def package_versions(stage: str) -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in RUNTIME}
    if {key: value.split("+")[0] for key, value in versions.items()} != RUNTIME:
        raise RuntimeError(f"GPU runtime differs from E39 pinned packages: {versions}")
    if stage != "reference":
        import io

        arrow_version = importlib.metadata.version("pyarrow")
        if not 15 <= int(arrow_version.split(".")[0]) < 30:
            raise RuntimeError(f"Kaggle GPU kernel pyarrow is outside the probe range: {arrow_version}")
        buffer = io.BytesIO()
        trial = pd.DataFrame({"id": ["schema-probe"], "model": ["schema-probe"]})
        trial.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)
        if not pd.read_parquet(buffer, engine="pyarrow").equals(trial):
            raise RuntimeError("Kaggle GPU kernel cannot round-trip Parquet")
        versions["pyarrow"] = arrow_version
    return versions


def check_hardware(stage: str) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA accelerator is required for E39 deployment probe")
    devices = []
    for index in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": info.name, "total_bytes": info.total_memory})
    if stage == "reference":
        if len(devices) != 1 or "3090" not in devices[0]["name"]:
            raise RuntimeError("The smoke reference must use one server RTX 3090")
    else:
        if (len(devices) != 2 or any("T4" not in row["name"] for row in devices)
                or any(not (14 * 2**30 <= row["total_bytes"] <= 17 * 2**30)
                       for row in devices)):
            raise RuntimeError(f"The deployment probe requires Kaggle T4 x2: {devices}")
    return {"devices": devices, "torch_cuda": torch.version.cuda}


def device_map(mode: str) -> dict:
    if mode == "single":
        return {"": 0}
    if mode != "dual":
        raise ValueError("Device mode must be single or dual")
    # Qwen3-8B-Base has 36 decoder layers. This fixed engineering map is used
    # only if the single T4 probe cannot load; it is never chosen by scores.
    mapping = {"model.embed_tokens": 0}
    mapping.update({f"model.layers.{index}": 0 if index < 18 else 1
                    for index in range(36)})
    mapping.update({"model.norm": 1, "score": 1})
    return mapping


def load_model(model_dir: Path, adapter_dir: Path, labels: tuple[str, ...], mode: str):
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise RuntimeError("E39 tokenizer lacks its pinned pad token")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        quantization_config=quantization, torch_dtype=torch.float16,
        device_map=device_map(mode),
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    return model, tokenizer


def sample_actual_gpu_free(minimum: list[int]) -> None:
    """Record driver-visible free bytes, including non-PyTorch allocations."""
    import torch

    for index in range(len(minimum)):
        free, _ = torch.cuda.mem_get_info(index)
        minimum[index] = min(minimum[index], free)


def score_texts(model, tokenizer, texts: list[str], labels: tuple[str, ...],
                minimum_actual_free: list[int],
                timeout_seconds: float | None = None) -> np.ndarray:
    import torch

    result = np.empty((len(texts), len(labels)), dtype=np.float32)
    began = time.monotonic()
    with torch.inference_mode():
        for index, text in enumerate(texts):
            if timeout_seconds is not None and time.monotonic() - began > timeout_seconds:
                raise TimeoutError("E39 4000-text GPU proxy exceeded three hours")
            encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt", add_special_tokens=True)
            encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
            logits = model(**encoded).logits
            if logits.shape != (1, len(labels)) or not torch.isfinite(logits).all():
                raise FloatingPointError(f"Invalid E39 logits at proxy row {index}")
            sample_actual_gpu_free(minimum_actual_free)
            result[index] = logits.float().cpu().numpy()[0]
            sample_actual_gpu_free(minimum_actual_free)
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite E39 proxy scores")
    return result


def memory_report(minimum_actual_free: list[int]) -> list[dict]:
    import torch

    rows = []
    for index in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(index).total_memory
        reserved = torch.cuda.max_memory_reserved(index)
        free, _ = torch.cuda.mem_get_info(index)
        rows.append({"index": index, "peak_reserved_bytes": reserved,
                     "total_bytes": total, "remaining_at_peak_bytes": total - reserved,
                     "current_actual_free_bytes": free,
                     "minimum_actual_free_bytes": min(free, minimum_actual_free[index])})
    return rows


def reference_rows(train_path: Path, e0_oof: Path) -> tuple[list[int], list[str]]:
    texts = common.train_texts_only(train_path)
    if common.sha256_file(e0_oof) != E0_OOF_SHA256:
        raise ValueError("E0 five-fold assignment file differs from the locked protocol")
    frame = pd.read_csv(train_path, usecols=["id", "text"], dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    if frame["text"].tolist() != texts or frame["id"].duplicated().any():
        raise ValueError("Official training row order changed")
    oof = pd.read_csv(e0_oof, usecols=["id", "fold"], dtype=str, keep_default_na=False)
    if len(oof) != 2160 or oof["id"].duplicated().any() or not oof["fold"].isin(
            ["1", "2", "3", "4", "5"]).all():
        raise ValueError("E0 development OOF split is invalid")
    training_ids = set(oof.loc[oof["fold"] != "1", "id"])
    positions = [index for index, item in enumerate(frame["id"]) if item in training_ids][:3]
    if tuple(positions) != SMOKE_TRAIN_ROW_INDICES:
        raise RuntimeError("Fold-1 training-side smoke rows differ from preregistration")
    return positions, [texts[index] for index in positions]


def run(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    versions = package_versions(args.stage)
    hardware = check_hardware(args.stage)
    manifest = read_manifest(args.manifest)
    labels = tuple(manifest["labels"])
    if args.stage == "environment":
        print(json.dumps({"stage": args.stage, "packages": versions,
                          "hardware": hardware, "internet_required": False}, indent=2), flush=True)
        return
    verify_assets(args.model_dir, args.adapter_dir, manifest)
    minimum_actual_free = [torch.cuda.mem_get_info(index)[0]
                           for index in range(torch.cuda.device_count())]
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    model, tokenizer = load_model(args.model_dir, args.adapter_dir, labels, args.device_mode)
    sample_actual_gpu_free(minimum_actual_free)
    loaded_seconds = time.monotonic() - started
    if args.stage == "reference":
        positions, texts = reference_rows(args.train, args.e0_oof)
        logits = score_texts(model, tokenizer, texts, labels, minimum_actual_free)
        if args.output.exists():
            raise FileExistsError(args.output)
        report = {"format_version": 1, "fold": 1, "labels": labels,
                  "model_revision": manifest["model_revision"],
                  "train_sha256": common.TRAIN_SHA256,
                  "base_manifest_sha256": common.sha256_file(args.manifest),
                  "adapter_weights_sha256": manifest["adapter_files"]["adapter_model.safetensors"],
                  "max_abs_logit_difference_allowed": SMOKE_ATOL,
                  "rows": [{"train_row_index": position,
                            "text_sha256": common.sha256_text(text),
                            "logits": logits[index].astype(float).tolist(),
                            "argmax": int(logits[index].argmax())}
                           for index, (position, text) in enumerate(zip(positions, texts, strict=True))],
                  "packages": versions, "hardware": hardware,
                  "memory": memory_report(minimum_actual_free),
                  "seconds": time.monotonic() - started}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("E39_TRAIN_SIDE_SMOKE_REFERENCE_READY", flush=True)
        return

    reference = json.loads(args.smoke_reference.read_text(encoding="utf-8"))
    if (reference.get("format_version") != 1 or reference.get("fold") != 1
            or reference.get("labels") != list(labels)
            or reference.get("train_sha256") != common.TRAIN_SHA256
            or reference.get("base_manifest_sha256") != common.sha256_file(args.manifest)
            or reference.get("adapter_weights_sha256") !=
            manifest["adapter_files"]["adapter_model.safetensors"]
            or reference.get("max_abs_logit_difference_allowed") != SMOKE_ATOL
            or len(reference.get("rows", [])) != 3):
        raise ValueError("E39 training-side smoke reference does not match the locked assets")
    source_texts = common.train_texts_only(args.train)
    selected = []
    for item in reference["rows"]:
        ordinal = item["train_row_index"]
        if not isinstance(ordinal, int) or not 0 <= ordinal < len(source_texts):
            raise ValueError("Smoke row ordinal is invalid")
        text = source_texts[ordinal]
        if common.sha256_text(text) != item["text_sha256"]:
            raise ValueError("Smoke train-side text hash differs")
        selected.append(text)
    actual = score_texts(model, tokenizer, selected, labels, minimum_actual_free)
    expected = np.asarray([item["logits"] for item in reference["rows"]], dtype=np.float32)
    if (expected.shape != (3, 18) or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, atol=SMOKE_ATOL, rtol=0)
            or not np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))):
        raise RuntimeError("E39 T4 smoke logits differ from 3090 fold-1 reference")
    if args.stage == "smoke":
        if args.output.exists():
            raise FileExistsError(args.output)
        report = {"stage": "smoke", "passed": True, "device_mode": args.device_mode,
                  "model_manifest_sha256": common.sha256_file(args.manifest),
                  "smoke_reference_sha256": common.sha256_file(args.smoke_reference),
                  "packages": versions, "hardware": hardware,
                  "load_seconds": loaded_seconds, "total_seconds": time.monotonic() - started,
                  "memory": memory_report(minimum_actual_free), "logit_atol": SMOKE_ATOL,
                  "argmax_equal": True}
        if any(row["minimum_actual_free_bytes"] < MIN_T4_MARGIN_BYTES
               for row in report["memory"]):
            raise MemoryError("E39 T4 smoke actual free memory fell below 256 MiB")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("E39_T4_SMOKE_PASS", flush=True)
        return

    if args.stage != "full":
        raise ValueError(args.stage)
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    if (smoke.get("stage") != "smoke" or smoke.get("passed") is not True
            or smoke.get("device_mode") != args.device_mode
            or smoke.get("model_manifest_sha256") != common.sha256_file(args.manifest)
            or smoke.get("smoke_reference_sha256") != common.sha256_file(args.smoke_reference)):
        raise ValueError("A matching successful T4 smoke report is required before full proxy")
    ids, texts = common.proxy_fixture(args.train)
    inference_started = time.monotonic()
    logits = score_texts(model, tokenizer, texts, labels, minimum_actual_free,
                         timeout_seconds=MAX_FULL_INFERENCE_SECONDS)
    inference_seconds = time.monotonic() - inference_started
    memory = memory_report(minimum_actual_free)
    if any(row["minimum_actual_free_bytes"] < MIN_T4_MARGIN_BYTES for row in memory):
        raise MemoryError("E39 4000-row T4 actual free memory fell below 256 MiB")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids, labels=np.asarray(labels, dtype=str),
                        logits=logits)
    summary = {"stage": "full_unscored_proxy", "rows": len(ids),
               "device_mode": args.device_mode,
               "model_manifest_sha256": common.sha256_file(args.manifest),
               "smoke_reference_sha256": common.sha256_file(args.smoke_reference),
               "packages": versions, "hardware": hardware, "memory": memory,
               "load_seconds": loaded_seconds, "inference_seconds": inference_seconds,
               "total_seconds": time.monotonic() - started,
               "validation_scores_computed": False,
               "labels_loaded": False}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
    print("E39_T4_FULL_PROXY_READY rows=4000", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["environment", "reference", "smoke", "full"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--e0-oof", type=Path)
    parser.add_argument("--smoke-reference", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-mode", choices=["single", "dual"], default="single")
    args = parser.parse_args()
    if args.stage != "environment":
        for name in ("model_dir", "adapter_dir", "train", "output"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required for {args.stage}")
    if args.stage == "reference" and args.e0_oof is None:
        parser.error("--e0-oof is required for reference")
    if args.stage in ("smoke", "full") and args.smoke_reference is None:
        parser.error("--smoke-reference is required")
    if args.stage == "full" and args.smoke_report is None:
        parser.error("--smoke-report is required for full")
    run(args)


if __name__ == "__main__":
    main()
