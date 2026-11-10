#!/usr/bin/env python3
"""Guarded final fit, one-time holdout check, and offline Kaggle submission.

The selected recipe is E0c plus corrected 12-epoch RoBERTa, using the exact
equal-rank rule from run_fixed_blend.py. Nothing runs on the holdout merely by
importing this module or displaying --help. See docs/final_pipeline.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_e0 import LABELS, load_and_validate, sha256
from run_fixed_blend import SCORE_COLUMNS, ranks


RECIPE = "e0c_plus_roberta_12ep_scalerfix_equal_rank"
ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
ROBERTA_WEIGHT_SHA256 = "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81"
PINNED_SNAPSHOT_SHA256 = {
    "model.safetensors": ROBERTA_WEIGHT_SHA256,
    "config.json": "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9",
    "tokenizer.json": "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    "tokenizer_config.json": "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    "vocab.json": "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}
REQUIRED_CONFIG = {
    "model_id": "FacebookAI/roberta-base",
    "requested_revision": ROBERTA_REVISION,
    "epochs": 12,
    "encoder_learning_rate": 2e-5,
    "head_learning_rate": 1e-3,
    "max_length": 256,
    "batch_size": 16,
    "seed": 42,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "pooling": "mean",
    "roberta_pooler_disabled": True,
}
SOURCE_FILES = ("final_pipeline.py", "run_e0.py", "run_style.py", "run_combined.py", "run_fixed_blend.py")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in SOURCE_FILES}


def check_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")


def check_config(config: dict) -> None:
    for key, expected in REQUIRED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"RoBERTa config {key!r} must be {expected!r}, got {config.get(key)!r}")
    if config.get("epoch_selection") != "fixed final epoch":
        raise ValueError("RoBERTa config must select the fixed final epoch")


def verify_pretrained_snapshot(path: Path) -> None:
    if path.name != ROBERTA_REVISION or path.parent.name != "snapshots":
        raise ValueError(f"Base checkpoint must be the pinned snapshot directory for {ROBERTA_REVISION}")
    found = {entry.name for entry in path.iterdir()}
    if found != set(PINNED_SNAPSHOT_SHA256):
        raise ValueError(f"Base snapshot files differ from pinned manifest: {sorted(found ^ set(PINNED_SNAPSHOT_SHA256))}")
    for name, expected in PINNED_SNAPSHOT_SHA256.items():
        if sha256(path / name) != expected:
            raise ValueError(f"Base snapshot hash differs from pinned revision: {name}")


def load_manifest(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"]:
        raise ValueError("Expected prompt_id,split in the saved E0 manifest")
    if manifest["prompt_id"].duplicated().any() or set(manifest["prompt_id"]) != set(frame["prompt_id"]):
        raise ValueError("Saved manifest does not cover every training prompt exactly once")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("Manifest must contain development and holdout splits")
    if manifest["split"].eq("holdout").sum() != 30 or manifest["split"].eq("development").sum() != 120:
        raise ValueError("Expected the original 120/30 prompt partition")
    return manifest


def partition(frame: pd.DataFrame, manifest: pd.DataFrame, name: str) -> pd.DataFrame:
    prompts = set(manifest.loc[manifest["split"].eq(name), "prompt_id"])
    return frame.loc[frame["prompt_id"].isin(prompts)].reset_index(drop=True)


def check_oof(path: Path, development: pd.DataFrame) -> pd.DataFrame:
    oof = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "prompt_id", "true_model", "fold", *SCORE_COLUMNS}
    if not required.issubset(oof.columns):
        raise ValueError(f"OOF score file lacks {sorted(required - set(oof.columns))}: {path}")
    expected = development[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if not oof[["id", "prompt_id", "true_model"]].equals(expected):
        raise ValueError(f"OOF rows do not match saved development rows: {path}")
    if set(oof["fold"]) != {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Unexpected folds in {path}")
    scores = oof[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite OOF scores in {path}")
    return oof


def check_e4_run(name: str, directory: Path, development: pd.DataFrame, expected_fold: pd.Series,
                 train_hash: str, manifest_hash: str) -> dict:
    from sklearn.metrics import balanced_accuracy_score
    metrics_path = directory / "metrics.json"
    oof_path = directory / "oof_predictions.csv"
    metrics = read_json(metrics_path)
    oof = pd.read_csv(oof_path, dtype=str, keep_default_na=False)
    if metrics.get("condition") != name or metrics.get("train_sha256") != train_hash or metrics.get("split_manifest_sha256") != manifest_hash:
        raise ValueError(f"E4 {name} metrics do not match the selected condition/data/split")
    if metrics.get("n_rows_development") != 2160 or metrics.get("n_prompts_development") != 120:
        raise ValueError(f"E4 {name} metrics have unexpected development counts")
    keys = ["id", "prompt_id", "true_model", "fold"]
    expected = development[["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if not set(keys + ["pred_model"]).issubset(oof.columns):
        raise ValueError(f"E4 {name} OOF lacks required columns")
    if not oof[["id", "prompt_id", "true_model"]].equals(expected) or not oof["fold"].equals(expected_fold):
        raise ValueError(f"E4 {name} OOF does not match the saved development rows/folds")
    if not set(oof["pred_model"]).issubset(LABELS):
        raise ValueError(f"E4 {name} OOF contains unexpected predictions")
    reproduced = float(balanced_accuracy_score(oof["true_model"], oof["pred_model"]))
    if not np.isclose(reproduced, metrics.get("oof_balanced_accuracy", np.nan), atol=1e-12):
        raise ValueError(f"E4 {name} OOF score does not reproduce its metrics")
    return {"balanced_accuracy": reproduced, "metrics_sha256": sha256(metrics_path), "oof_sha256": sha256(oof_path)}


def create_lock(args: argparse.Namespace) -> None:
    """Run only after E4 is complete and the candidate is frozen."""
    if args.output.exists():
        raise ValueError(f"Selection lock already exists: {args.output}")
    if len(args.decision_note.strip()) < 20:
        raise ValueError("A substantive --decision-note is required before locking selection")
    frame = load_and_validate(args.train)
    manifest = load_manifest(args.manifest, frame)
    development = partition(frame, manifest, "development")
    if len(development) != 2160:
        raise ValueError("Expected 2,160 development rows")
    e0c_oof = check_oof(args.e0c_oof, development)
    roberta_oof = check_oof(args.roberta_oof, development)
    if not e0c_oof["fold"].equals(roberta_oof["fold"]):
        raise ValueError("The two selected OOF files have different fold assignments")
    blend_oof = pd.read_csv(args.blend_oof, dtype=str, keep_default_na=False)
    expected_columns = ["id", "prompt_id", "true_model", "fold", "pred_model"]
    if not set(expected_columns).issubset(blend_oof.columns):
        raise ValueError("Saved corrected E7e-R blend OOF lacks required columns")
    if not blend_oof[["id", "prompt_id", "true_model", "fold"]].equals(e0c_oof[["id", "prompt_id", "true_model", "fold"]]):
        raise ValueError("Saved corrected E7e-R blend OOF rows do not match the selected component scores")
    reproduced = equal_rank_predict(
        e0c_oof[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64),
        roberta_oof[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64),
    )
    if not np.array_equal(reproduced, blend_oof["pred_model"].to_numpy()):
        raise ValueError("Fixed rank rule does not reproduce the saved corrected E7e-R OOF predictions")
    config = read_json(args.roberta_config)
    check_config(config)
    if config.get("train_sha256") != sha256(args.train) or config.get("split_manifest_sha256") != sha256(args.manifest):
        raise ValueError("Corrected RoBERTa run was not based on this train file and manifest")
    e4 = read_json(args.e4_comparison)
    e4_config = read_json(args.e4_config)
    e4_cross_config = read_json(args.e4_cross_config)
    for key in ("reference_balanced_accuracy", "candidate_balanced_accuracy", "difference", "n_prompts"):
        if key not in e4:
            raise ValueError(f"E4 comparison lacks {key}")
    if e4["n_prompts"] != 120:
        raise ValueError("E4 comparison must cover the 120 development prompts")
    if e4_config.get("train_sha256") != sha256(args.train) or e4_config.get("split_manifest_sha256") != sha256(args.manifest):
        raise ValueError("E4 run was not based on the locked train file and manifest")
    if e4_config.get("holdout_prompts_untouched") != 30:
        raise ValueError("E4 config does not confirm the 30 untouched holdout prompts")
    if e4_cross_config.get("train_sha256") != sha256(args.train) or e4_cross_config.get("split_manifest_sha256") != sha256(args.manifest):
        raise ValueError("E4 cross-prompt run does not match the locked train file and manifest")
    if e4_cross_config.get("holdout_prompts_untouched") != 30 or e4_cross_config.get("selected_condition") != "ce_supcon_crossprompt":
        raise ValueError("E4 cross-prompt run config is incomplete or selected a different condition")
    e4_runs = {
        "ce_only": check_e4_run("ce_only", args.e4_dir / "ce_only", development, e0c_oof["fold"], sha256(args.train), sha256(args.manifest)),
        "ce_supcon_crossprompt": check_e4_run("ce_supcon_crossprompt", args.e4_cross_dir / "ce_supcon_crossprompt", development, e0c_oof["fold"], sha256(args.train), sha256(args.manifest)),
        "ce_supcon": check_e4_run("ce_supcon", args.e4_dir / "ce_supcon", development, e0c_oof["fold"], sha256(args.train), sha256(args.manifest)),
    }
    if not np.isclose(e4["reference_balanced_accuracy"], e4_runs["ce_only"]["balanced_accuracy"], atol=1e-12) or not np.isclose(e4["candidate_balanced_accuracy"], e4_runs["ce_supcon"]["balanced_accuracy"], atol=1e-12):
        raise ValueError("E4 original comparison does not match completed condition metrics")
    lock = {
        "recipe": RECIPE,
        "selection_frozen": True,
        "e4_completed": True,
        "decision_note": args.decision_note.strip(),
        "train_sha256": sha256(args.train),
        "manifest_sha256": sha256(args.manifest),
        "e0c_oof_sha256": sha256(args.e0c_oof),
        "roberta_oof_sha256": sha256(args.roberta_oof),
        "blend_oof_sha256": sha256(args.blend_oof),
        "roberta_config_sha256": sha256(args.roberta_config),
        "e4_comparison_sha256": sha256(args.e4_comparison),
        "e4_config_sha256": sha256(args.e4_config),
        "e4_cross_config_sha256": sha256(args.e4_cross_config),
        "e4_runs": e4_runs,
        "roberta_config": {key: config[key] for key in REQUIRED_CONFIG},
        "labels": list(LABELS),
        "rank_rule": "sum row-wise ascending ranks; E0c breaks exact ties by 1e-4 times its rank",
        "holdout_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, lock)
    print(f"Selection locked: {args.output}")


def verify_lock(lock_path: Path, train_path: Path, manifest_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    lock = read_json(lock_path)
    if lock.get("recipe") != RECIPE or lock.get("selection_frozen") is not True or lock.get("e4_completed") is not True:
        raise ValueError("Selection lock is missing or does not specify the frozen E0c + corrected RoBERTa recipe")
    if lock.get("labels") != list(LABELS):
        raise ValueError("Selection lock labels differ from the official label order")
    if sha256(train_path) != lock.get("train_sha256") or sha256(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("Training data or split manifest changed after selection lock")
    check_config({**lock.get("roberta_config", {}), "epoch_selection": "fixed final epoch"})
    frame = load_and_validate(train_path)
    manifest = load_manifest(manifest_path, frame)
    return lock, frame, manifest


def equal_rank_predict(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != len(LABELS):
        raise ValueError("Both score matrices must have one column per official label")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Decision scores must be finite")
    left_ranks = ranks(left)
    selected = (left_ranks + ranks(right) + left_ranks * 1e-4).argmax(axis=1)
    return np.asarray(LABELS, dtype=object)[selected]


def load_runtime():
    # Keep CLI help, lock creation, and lightweight checks usable without a GPU.
    import joblib
    import sklearn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import transformers
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    class TextDataset(Dataset):
        def __init__(self, encoded, labels=None):
            self.encoded = encoded
            self.labels = labels

        def __len__(self):
            return len(self.encoded["input_ids"])

        def __getitem__(self, index):
            row = {key: self.encoded[key][index] for key in ("input_ids", "attention_mask")}
            if self.labels is not None:
                row["labels"] = int(self.labels[index])
            return row

    class Classifier(nn.Module):
        def __init__(self, config, pretrained_dir=None):
            super().__init__()
            if pretrained_dir is None:
                self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
            else:
                self.encoder = AutoModel.from_pretrained(pretrained_dir, local_files_only=True, add_pooling_layer=False)
            self.dropout = nn.Dropout(REQUIRED_CONFIG["dropout"])
            self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

        def forward(self, input_ids, attention_mask):
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).to(encoded.last_hidden_state.dtype)
            pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return self.classifier(self.dropout(pooled))

    return locals()


def seed_all(torch, seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_texts(tokenizer, texts: list[str], max_length: int) -> dict:
    value = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return {key: value[key] for key in ("input_ids", "attention_mask")}


def score_neural(runtime, model, tokenizer, texts: list[str], batch_size: int, device) -> np.ndarray:
    torch = runtime["torch"]
    loader = runtime["DataLoader"](
        runtime["TextDataset"](encode_texts(tokenizer, texts, REQUIRED_CONFIG["max_length"])),
        batch_size=batch_size, shuffle=False,
        collate_fn=lambda rows: tokenizer.pad(rows, padding=True, return_tensors="pt"),
    )
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.empty((0, len(LABELS)), dtype=np.float32)


def fit(args: argparse.Namespace, mode: str) -> None:
    check_empty_output(args.output_dir)
    lock, frame, manifest = verify_lock(args.selection_lock, args.train, args.manifest)
    if mode == "full":
        report = read_json(args.holdout_report)
        if report.get("selection_lock_sha256") != sha256(args.selection_lock) or report.get("recipe") != RECIPE:
            raise ValueError("Full fit requires the one-time holdout report for this selection lock")
        if report.get("n_rows_holdout") != 539:
            raise ValueError("Holdout report has unexpected row count")
        train_frame = frame.reset_index(drop=True)
    else:
        train_frame = partition(frame, manifest, "development")
        if len(train_frame) != 2160:
            raise ValueError("Expected 2,160 development rows")
    verify_pretrained_snapshot(args.pretrained_dir)
    runtime = load_runtime()
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("Final RoBERTa fit requires a CUDA device")
    from run_combined import make_pipeline
    start = time.perf_counter()
    texts = train_frame["text"].to_numpy()
    labels = train_frame["model"].to_numpy()
    surface = make_pipeline()
    surface.fit(texts, labels)
    seed_all(torch, REQUIRED_CONFIG["seed"])
    tokenizer = runtime["AutoTokenizer"].from_pretrained(args.pretrained_dir, local_files_only=True, use_fast=True)
    encoded = encode_texts(tokenizer, texts.tolist(), REQUIRED_CONFIG["max_length"])
    numerical_labels = np.asarray([LABELS.index(label) for label in labels], dtype=np.int64)
    dataset = runtime["TextDataset"](encoded, numerical_labels)
    generator = torch.Generator().manual_seed(REQUIRED_CONFIG["seed"])

    def collate(rows):
        padded = tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in rows],
            padding=True, return_tensors="pt",
        )
        padded["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        return padded

    loader = runtime["DataLoader"](
        dataset, batch_size=REQUIRED_CONFIG["batch_size"], shuffle=True,
        generator=generator, collate_fn=collate, num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda:0")
    config = runtime["AutoConfig"].from_pretrained(args.pretrained_dir, local_files_only=True)
    if config.model_type != "roberta" or config.hidden_size != 768:
        raise ValueError("Local pretrained config is not RoBERTa base")
    model = runtime["Classifier"](config, args.pretrained_dir).to(device)
    class_counts = np.bincount(numerical_labels, minlength=len(LABELS))
    if (class_counts == 0).any():
        raise ValueError("A training class is missing")
    class_weights = torch.tensor(
        len(dataset) / (len(LABELS) * class_counts.astype(np.float32)),
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": REQUIRED_CONFIG["encoder_learning_rate"]},
        {"params": model.classifier.parameters(), "lr": REQUIRED_CONFIG["head_learning_rate"]},
    ], weight_decay=REQUIRED_CONFIG["weight_decay"])
    total_steps = REQUIRED_CONFIG["epochs"] * len(loader)
    scheduler = runtime["get_linear_schedule_with_warmup"](
        optimizer, num_warmup_steps=round(REQUIRED_CONFIG["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses = []
    skipped = 0
    for epoch in range(1, REQUIRED_CONFIG["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = runtime["F"].cross_entropy(logits.float(), batch["labels"].to(device), weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= before:
                scheduler.step()
            else:
                skipped += 1
            loss_sum += float(loss.item()) * len(batch["labels"])
        losses.append(loss_sum / len(dataset))
        print(f"{mode} epoch {epoch}/12 loss={losses[-1]:.6f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime["joblib"].dump(surface, args.output_dir / "e0c.joblib", compress=3)
    torch.save(model.cpu().state_dict(), args.output_dir / "roberta_state_dict.pt")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    config.save_pretrained(args.output_dir / "encoder_config")
    metadata = {
        "recipe": RECIPE,
        "mode": mode,
        "selection_lock_sha256": sha256(args.selection_lock),
        "train_sha256": sha256(args.train),
        "manifest_sha256": sha256(args.manifest),
        "n_training_rows": len(train_frame),
        "n_training_prompts": int(train_frame["prompt_id"].nunique()),
        "roberta_config": REQUIRED_CONFIG,
        "roberta_weight_sha256": ROBERTA_WEIGHT_SHA256,
        "e0c_joblib_sha256": sha256(args.output_dir / "e0c.joblib"),
        "roberta_state_sha256": sha256(args.output_dir / "roberta_state_dict.pt"),
        "source_sha256": source_hashes(),
        "sklearn_version": runtime["sklearn"].__version__,
        "torch_version": torch.__version__,
        "transformers_version": runtime["transformers"].__version__,
        "epoch_train_loss": losses,
        "skipped_optimizer_steps": skipped,
        "fit_seconds": time.perf_counter() - start,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(f"Fitted {mode} package: {args.output_dir}")


def load_package(path: Path, expected_mode: str):
    runtime = load_runtime()
    metadata = read_json(path / "metadata.json")
    if metadata.get("recipe") != RECIPE or metadata.get("mode") != expected_mode:
        raise ValueError(f"Expected a {expected_mode} package for the fixed recipe")
    if metadata.get("sklearn_version") != runtime["sklearn"].__version__:
        raise ValueError("scikit-learn version differs from the fitted joblib package")
    if metadata.get("source_sha256") != source_hashes():
        raise ValueError("Inference source files differ from those used for fitting")
    if metadata.get("torch_version", "").split("+")[0] != runtime["torch"].__version__.split("+")[0] or metadata.get("transformers_version") != runtime["transformers"].__version__:
        raise ValueError("torch or transformers version differs from the fitted package")
    for name, key in (("e0c.joblib", "e0c_joblib_sha256"), ("roberta_state_dict.pt", "roberta_state_sha256")):
        if sha256(path / name) != metadata.get(key):
            raise ValueError(f"Model artifact changed after fitting: {name}")
    surface = runtime["joblib"].load(path / "e0c.joblib")
    tokenizer = runtime["AutoTokenizer"].from_pretrained(path / "tokenizer", local_files_only=True, use_fast=True)
    config = runtime["AutoConfig"].from_pretrained(path / "encoder_config", local_files_only=True)
    model = runtime["Classifier"](config)
    state = runtime["torch"].load(path / "roberta_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = runtime["torch"].device("cuda:0" if runtime["torch"].cuda.is_available() else "cpu")
    model.to(device)
    return runtime, metadata, surface, tokenizer, model, device


def predict_package(path: Path, texts: list[str], mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runtime, metadata, surface, tokenizer, model, device = load_package(path, mode)
    left = surface.decision_function(texts)
    if list(surface.named_steps["classifier"].classes_) != sorted(LABELS):
        raise ValueError("E0c classifier class set/order is unexpected")
    # LinearSVC uses lexical class order; reorder to the official fixed order.
    positions = [list(surface.named_steps["classifier"].classes_).index(label) for label in LABELS]
    left = np.asarray(left[:, positions], dtype=np.float64)
    right = score_neural(runtime, model, tokenizer, texts, 16, device).astype(np.float64)
    return equal_rank_predict(left, right), left, right


def evaluate_holdout(args: argparse.Namespace) -> None:
    if args.report.resolve() != args.selection_lock.with_name("holdout_report.json").resolve():
        raise ValueError("The one-time holdout report must be beside the selection lock")
    if args.report.exists():
        raise ValueError(f"Holdout report already exists; one-time evaluation is locked: {args.report}")
    lock, frame, manifest = verify_lock(args.selection_lock, args.train, args.manifest)
    holdout = partition(frame, manifest, "holdout")
    if len(holdout) != 539:
        raise ValueError("Expected the original 539 holdout rows")
    package_meta = read_json(args.dev_package / "metadata.json")
    if package_meta.get("selection_lock_sha256") != sha256(args.selection_lock) or package_meta.get("train_sha256") != lock["train_sha256"]:
        raise ValueError("Development package does not match this locked selection")
    if package_meta.get("n_training_rows") != 2160 or package_meta.get("n_training_prompts") != 120:
        raise ValueError("Development package contains unexpected training rows")
    predicted, left, right = predict_package(args.dev_package, holdout["text"].tolist(), "development")
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    true = holdout["model"].to_numpy()
    confusion = confusion_matrix(true, predicted, labels=LABELS)
    report = {
        "recipe": RECIPE,
        "selection_lock_sha256": sha256(args.selection_lock),
        "dev_package_metadata_sha256": sha256(args.dev_package / "metadata.json"),
        "n_rows_holdout": len(holdout),
        "n_prompts_holdout": int(holdout["prompt_id"].nunique()),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "e0c_balanced_accuracy": float(balanced_accuracy_score(true, np.asarray(LABELS)[left.argmax(axis=1)])),
        "roberta_balanced_accuracy": float(balanced_accuracy_score(true, np.asarray(LABELS)[right.argmax(axis=1)])),
        "labels": list(LABELS),
        "confusion_matrix": confusion.tolist(),
        "class_recall": (confusion.diagonal() / confusion.sum(axis=1)).tolist(),
        "note": "One-time locked holdout estimate; do not retune on these labels.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"One-time holdout balanced accuracy: {report['balanced_accuracy']:.6f}")
    print(f"Report: {args.report}")


def load_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(frame.columns) != ["id", "text"] or frame.empty:
        raise ValueError("Test CSV must have exactly id,text columns and at least one row")
    if frame["id"].duplicated().any() or frame["id"].str.strip().eq("").any() or frame["text"].str.strip().eq("").any():
        raise ValueError("Test IDs must be unique and both fields nonempty")
    return frame


def submit(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(f"Submission already exists: {args.output}")
    test = load_test(args.test)
    predicted, _, _ = predict_package(args.full_package, test["text"].tolist(), "full")
    if len(predicted) != len(test) or not set(predicted).issubset(LABELS):
        raise RuntimeError("Prediction count or labels are invalid")
    output = pd.DataFrame({"id": test["id"], "model": predicted})
    if args.sample_submission is not None:
        sample = pd.read_parquet(args.sample_submission)
        if list(sample.columns) != ["id", "model"] or not sample["id"].astype(str).equals(output["id"]):
            raise ValueError("Sample submission schema or ID order differs from test.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, engine="pyarrow")
    roundtrip = pd.read_parquet(args.output, engine="pyarrow")
    if not roundtrip.equals(output):
        raise RuntimeError("Parquet round-trip changed IDs or predictions")
    print(f"Offline submission written: {args.output} ({len(output)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-selection", help="Record the frozen recipe after E4 concludes")
    freeze.add_argument("--train", type=Path, default=Path("data/train.csv"))
    freeze.add_argument("--manifest", type=Path, default=Path("outputs/e0_tfidf_svm/split_manifest.csv"))
    freeze.add_argument("--e0c-oof", type=Path, default=Path("outputs/e0c_tfidf_plus_style/decision_scores.csv"))
    freeze.add_argument("--roberta-oof", type=Path, default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"))
    freeze.add_argument("--blend-oof", type=Path, default=Path("outputs/e7_blend_e0c_roberta_ft_scalerfix/oof_predictions.csv"))
    freeze.add_argument("--roberta-config", type=Path, default=Path("outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/run_config.json"))
    freeze.add_argument("--e4-comparison", type=Path, default=Path("outputs/e4_prompt_contrastive/comparison.json"))
    freeze.add_argument("--e4-config", type=Path, default=Path("outputs/e4_prompt_contrastive/run_config.json"))
    freeze.add_argument("--e4-cross-config", type=Path, default=Path("outputs/e4_prompt_contrastive_crossprompt/run_config.json"))
    freeze.add_argument("--e4-dir", type=Path, default=Path("outputs/e4_prompt_contrastive"))
    freeze.add_argument("--e4-cross-dir", type=Path, default=Path("outputs/e4_prompt_contrastive_crossprompt"))
    freeze.add_argument("--decision-note", required=True)
    freeze.add_argument("--output", type=Path, default=Path("outputs/final/selection_lock.json"))
    for name in ("fit-development", "fit-full"):
        command = commands.add_parser(name, help="Fit both selected models from the pinned base checkpoint")
        command.add_argument("--train", type=Path, default=Path("data/train.csv"))
        command.add_argument("--manifest", type=Path, default=Path("outputs/e0_tfidf_svm/split_manifest.csv"))
        command.add_argument("--selection-lock", type=Path, default=Path("outputs/final/selection_lock.json"))
        command.add_argument("--pretrained-dir", type=Path, required=True, help="Offline RoBERTa base snapshot with verified model.safetensors")
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "fit-full":
            command.add_argument("--holdout-report", type=Path, default=Path("outputs/final/holdout_report.json"))
    holdout = commands.add_parser("evaluate-holdout", help="Score locked holdout once; never use it for model selection")
    holdout.add_argument("--train", type=Path, default=Path("data/train.csv"))
    holdout.add_argument("--manifest", type=Path, default=Path("outputs/e0_tfidf_svm/split_manifest.csv"))
    holdout.add_argument("--selection-lock", type=Path, default=Path("outputs/final/selection_lock.json"))
    holdout.add_argument("--dev-package", type=Path, required=True)
    holdout.add_argument("--report", type=Path, default=Path("outputs/final/holdout_report.json"))
    submission = commands.add_parser("submit", help="Offline inference and validated submission.parquet")
    submission.add_argument("--full-package", type=Path, required=True)
    submission.add_argument("--test", type=Path, default=Path("data/test.csv"))
    submission.add_argument("--sample-submission", type=Path, default=None)
    submission.add_argument("--output", type=Path, default=Path("submission.parquet"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "freeze-selection":
            create_lock(args)
        elif args.command == "fit-development":
            fit(args, "development")
        elif args.command == "fit-full":
            fit(args, "full")
        elif args.command == "evaluate-holdout":
            evaluate_holdout(args)
        elif args.command == "submit":
            submit(args)
    except (OSError, ValueError, RuntimeError, ImportError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
