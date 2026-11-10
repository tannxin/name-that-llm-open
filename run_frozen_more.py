#!/usr/bin/env python3
"""E1 frozen encoder comparisons on E0's untouched prompt holdout protocol.

RoBERTa and microsoft/mpnet-base are raw pretrained encoder diagnostics; their
token states were not trained as sentence embeddings. All models use the same
256-token first-round cap (or an explicit 512-token ablation) and a fixed
linear classifier on development data only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from run_e0 import RANDOM_STATE, sha256
from run_frozen_embeddings import (
    ModelSpec,
    evaluate,
    get_embeddings,
    load_development,
    resolve_device,
)


MODELS = {
    "all-mpnet-base-v2": ModelSpec(
        slug="all-mpnet-base-v2",
        repo_id="sentence-transformers/all-mpnet-base-v2",
        max_length=256,
        prefix="",
        card_max_length=384,
    ),
    "roberta-base-raw-mlm": ModelSpec(
        slug="roberta-base-raw-mlm",
        repo_id="FacebookAI/roberta-base",
        max_length=256,
        prefix="",
        card_max_length=512,
    ),
    "style-embedding": ModelSpec(
        slug="style-embedding",
        repo_id="AnnaWegmann/Style-Embedding",
        max_length=256,
        prefix="",
        card_max_length=512,
    ),
    "mpnet-base-raw": ModelSpec(
        slug="mpnet-base-raw",
        repo_id="microsoft/mpnet-base",
        max_length=256,
        prefix="",
        card_max_length=512,
    ),
}

# Resolved from the official Hugging Face model API on 2026-11-04. Pinning the
# commits also permits a mirror endpoint when direct downloads are unavailable.
PINNED_REVISIONS = {
    "all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "roberta-base-raw-mlm": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    "style-embedding": "d7d0f5ca829316a8f5695e49dfce80b86db5e76c",
    "mpnet-base-raw": "6996ce1e91bd2a9c7d7f61daec37463394f73f09",
}
REPOSITORY_LICENSES = {
    "all-mpnet-base-v2": "apache-2.0",
    "roberta-base-raw-mlm": "mit",
    "style-embedding": None,
    "mpnet-base-raw": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("outputs/e0_tfidf_svm"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e1_frozen_more"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-length", type=int, choices=[256, 512], default=256,
        help="Token cap; 512 requires a separate --output-dir and a model that supports it",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device")
    return parser.parse_args()


def verify_e0_inputs(e0_dir: Path, train_sha: str) -> int:
    metrics = json.loads((e0_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("train_sha256") != train_sha:
        raise ValueError("Training CSV hash differs from the saved E0 baseline")
    if metrics.get("n_prompts_holdout") != 30 or metrics.get("n_folds") != 5:
        raise ValueError("E0 baseline is not the agreed 30-prompt, five-fold run")
    seed = metrics.get("random_state")
    if seed != RANDOM_STATE:
        raise ValueError(f"E0 seed {seed!r} differs from the agreed {RANDOM_STATE}")
    return seed


def annotate_metrics(model_dir: Path, slug: str) -> None:
    path = model_dir / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["comparison_batch"] = "e1_frozen_more"
    metrics["encoder_role"] = {
        "roberta-base-raw-mlm": "raw_mlm_diagnostic",
        "mpnet-base-raw": "raw_pretrained_encoder_diagnostic",
    }.get(slug, "sentence_embedding")
    metrics["encoder_pooling_rationale"] = (
        "predefined attention-masked mean of raw encoder token states"
        if slug in {"roberta-base-raw-mlm", "mpnet-base-raw"}
        else "model card attention-masked mean pooling"
    )
    metrics["text_case_preserved"] = True
    metrics["common_first_round_token_cap"] = 256
    metrics["token_cap_ablation"] = metrics["encoder_max_length"] != 256
    metrics["model_metadata_endpoint"] = os.environ.get(
        "HF_ENDPOINT", "https://huggingface.co"
    )
    metrics["weight_transport_url"] = os.environ.get("E1_WEIGHT_SOURCE_URL")
    metrics["verified_weight_sha256"] = os.environ.get("E1_WEIGHT_SHA256")
    metrics["repository_license_at_run"] = REPOSITORY_LICENSES[slug]
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_length != 256 and args.output_dir == Path("outputs/e1_frozen_more"):
        raise ValueError("512-token ablation requires a separate --output-dir")
    train_sha = sha256(args.train)
    seed = verify_e0_inputs(args.e0_dir, train_sha)
    frame, development, e0_oof = load_development(args.train, args.e0_dir)
    device = resolve_device(args.device)
    print(f"Development: {len(development)} rows; device={device}", flush=True)
    for slug in args.models:
        spec = MODELS[slug]
        if args.max_length > spec.card_max_length:
            raise ValueError(
                f"{slug} supports at most {spec.card_max_length} tokens in this comparison"
            )
        if args.max_length != spec.max_length:
            spec = replace(spec, max_length=args.max_length)
        revision = PINNED_REVISIONS[slug]
        model_dir = args.output_dir / slug
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
            seed,
        )
        annotate_metrics(model_dir, slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
