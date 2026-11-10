#!/usr/bin/env python3
"""Queue locked E30-A after the pinned download and an explicit E28 GPU release.

This is an operator-launched, one-shot server watcher. It does not start until
``--run`` is supplied. The release marker must be written by the E28 operator
after its GPU2 work is finished; this script never creates that marker.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from download_e30_qwen3_embedding_mirror import EXPECTED, HF_REVISION, MODEL_ID, MS_REVISION
from run_e30_supervised_embedding8b import (
    CONFIG_PATH,
    EXPECTED_SHA256,
    config,
    lock_input,
    sha256,
    verify_model_dir,
)


ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "outputs/e30_mirror_download"
MODEL_DIR = ROOT / "data/models/qwen3-embedding-8b-1d8ad4c"
MANIFEST = MODEL_DIR / "e30_mirror_manifest.json"
OUTPUT_DIR = ROOT / "outputs/e30_qwen3_embedding8b_v3"
RELEASE_MARKER = ROOT / "outputs/e28_qwen8b_fivefold/gpu2_released.ok"
RELEASE_CONTENT = "E28_GPU2_RELEASED_FOR_E30"
QUEUE_LOG_DIR = ROOT / "outputs/e30_gpu_queue"
POLL_SECONDS = 60
WAIT_SECONDS = 48 * 3600
STAGE_IDLE_WAIT_SECONDS = 30 * 60


def say(*parts: object) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), *parts, flush=True)


def acquire_lock() -> None:
    import fcntl

    QUEUE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = (QUEUE_LOG_DIR / "watcher.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    except Exception:
        handle.close()
        raise
    atexit.register(handle.close)


def protocol_check() -> dict:
    cfg = config()
    args = argparse.Namespace(
        train=ROOT / "data/train.csv",
        e0_dir=ROOT / "outputs/e0_tfidf_svm",
        output_dir=OUTPUT_DIR,
    )
    lock_input(args, cfg)
    return cfg


def download_complete() -> bool:
    # A resumed downloader writes its own log, so the original log may never
    # receive COMPLETE. The downloader writes this manifest only after all
    # pinned files pass SHA-256; verify_manifest_and_files rehashes them here.
    return MANIFEST.is_file()


def release_recorded() -> bool:
    return RELEASE_MARKER.is_file() and RELEASE_MARKER.read_text(encoding="utf-8").strip() == RELEASE_CONTENT


def verify_manifest_and_files(cfg: dict) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("model_id") != MODEL_ID
            or manifest.get("hf_revision") != HF_REVISION
            or manifest.get("modelscope_revision") != MS_REVISION
            or manifest.get("all_sha256_verified") is not True
            or cfg["model_revision"] != HF_REVISION):
        raise ValueError("E30 pinned mirror manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED):
        raise ValueError("E30 pinned mirror manifest file set differs")
    for name, (size, digest) in EXPECTED.items():
        if files[name] != {"size": size, "sha256": digest} or EXPECTED_SHA256[name] != digest:
            raise ValueError(f"E30 pinned mirror manifest differs for {name}")
        if (MODEL_DIR / name).stat().st_size != size:
            raise ValueError(f"E30 pinned mirror size differs for {name}")
    # Rehash every required file before taking a GPU. The model's own resource
    # and fit-fold entry points rehash them again before CUDA model loading.
    observed = verify_model_dir(MODEL_DIR, cfg)
    if set(observed) != set(EXPECTED):
        raise ValueError("E30 local model verification file set differs")
    say("HASH_CHECK_PASS", len(observed), "required files")


def gpu2_idle() -> str | None:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [[value.strip() for value in row] for row in csv.reader(query.stdout.splitlines()) if row]
    gpu2 = [row for row in rows if len(row) == 5 and row[0] == "2"]
    if len(gpu2) != 1 or "RTX 3090" not in gpu2[0][2]:
        raise RuntimeError("Cannot identify the expected physical RTX 3090 GPU2")
    _, uuid, _, memory_text, utilization_text = gpu2[0]
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
        raise RuntimeError("GPU2 UUID format changed")
    if int(memory_text) > 256 or int(utilization_text) != 0:
        return None
    processes = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    if processes.stdout.strip():
        return None
    return uuid


def wait_for_ready(cfg: dict) -> str:
    deadline = time.monotonic() + WAIT_SECONDS
    last_heartbeat = 0.0
    verified = False
    while time.monotonic() < deadline:
        if download_complete() and not verified:
            say("DOWNLOAD_COMPLETE_SIGNAL")
            verify_manifest_and_files(cfg)
            verified = True
        if verified and release_recorded():
            uuid = gpu2_idle()
            if uuid is not None:
                say("GPU2_RELEASED_AND_IDLE", uuid)
                return uuid
        if time.monotonic() - last_heartbeat >= 600:
            say("WAITING", "download_verified", verified,
                "e28_release_marker", release_recorded())
            last_heartbeat = time.monotonic()
        time.sleep(POLL_SECONDS)
    raise TimeoutError("E30 download, E28 release, or GPU2 idle wait exceeded 48 hours")


def wait_for_stage_gpu2(expected_uuid: str) -> None:
    deadline = time.monotonic() + STAGE_IDLE_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not release_recorded():
            raise RuntimeError("E28 GPU2 release marker disappeared or changed")
        observed_uuid = gpu2_idle()
        if observed_uuid == expected_uuid:
            return
        if observed_uuid is not None:
            raise RuntimeError("Physical GPU2 UUID changed while E30 was queued")
        time.sleep(5)
    raise TimeoutError("GPU2 did not become idle within 30 minutes before E30 stage")


def verify_resource_complete(cfg: dict) -> dict:
    path = OUTPUT_DIR / "resource_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("experiment") != cfg["experiment"]
            or report.get("stage") != "one_fold_resource_only"
            or report.get("config_sha256") != sha256(CONFIG_PATH)
            or report.get("protocol_lock_sha256") != sha256(OUTPUT_DIR / "protocol_lock.json")
            or report.get("model_revision") != HF_REVISION
            or report.get("model_file_sha256") != EXPECTED_SHA256
            or report.get("gate_passed") is not True
            or report.get("validation_accuracy_computed") is not False):
        raise RuntimeError("Existing E30 resource report is incomplete or failed its locked gate")
    resource_dir = OUTPUT_DIR / "resource_one_epoch"
    for artifact in (resource_dir / "adapter/adapter_config.json",
                     resource_dir / "adapter/adapter_model.safetensors",
                     resource_dir / "head.safetensors",
                     resource_dir / "tokenizer/tokenizer.json"):
        if not artifact.is_file():
            raise RuntimeError(f"Existing E30 resource artifact is missing: {artifact}")
    return report


def verify_fold_complete(cfg: dict, fold: int) -> None:
    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    path = fold_dir / "oof_predictions.csv"
    run = json.loads((fold_dir / "run.json").read_text(encoding="utf-8"))
    expected = {
        "experiment": cfg["experiment"], "fold": fold,
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_lock_sha256": sha256(OUTPUT_DIR / "protocol_lock.json"),
        "resource_report_sha256": sha256(OUTPUT_DIR / "resource_report.json"),
        "model_revision": HF_REVISION, "model_file_sha256": EXPECTED_SHA256,
        "epochs": cfg["epochs"], "train_rows": 1728, "validation_rows": 432,
        "oof_predictions_sha256": sha256(path),
        "no_epoch_or_threshold_selection": True, "holdout_evaluated": False,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Existing E30 fold {fold} is incomplete or differs from the locked run")
    for artifact in (fold_dir / "adapter/adapter_config.json",
                     fold_dir / "adapter/adapter_model.safetensors",
                     fold_dir / "head.safetensors",
                     fold_dir / "tokenizer/tokenizer.json"):
        if not artifact.is_file():
            raise RuntimeError(f"Existing E30 fold {fold} artifact is missing: {artifact}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 432 or {row.get("fold") for row in rows} != {str(fold)} or len({row.get("id") for row in rows}) != 432:
        raise RuntimeError(f"Existing E30 fold {fold} OOF identity is incomplete")


def run_stage(stage: str, cfg: dict, env: dict[str, str], *, fold: int | None = None) -> None:
    if stage == "resource" and (OUTPUT_DIR / "resource_report.json").exists():
        verify_resource_complete(cfg)
        say("SKIP_COMPLETE", "resource")
        return
    if stage == "fit-fold" and (OUTPUT_DIR / f"fold_{fold}").exists():
        verify_fold_complete(cfg, fold)
        say("SKIP_COMPLETE", f"fold_{fold}")
        return
    wait_for_stage_gpu2(env["CUDA_VISIBLE_DEVICES"])
    command = [
        sys.executable, "-u", str(ROOT / "run_e30_supervised_embedding8b.py"),
        stage, "--train", str(ROOT / "data/train.csv"),
        "--e0-dir", str(ROOT / "outputs/e0_tfidf_svm"),
        "--output-dir", str(OUTPUT_DIR),
    ]
    if stage in ("resource", "fit-fold"):
        command.extend(["--model-dir", str(MODEL_DIR)])
    if fold is not None:
        command.extend(["--fold", str(fold)])
    name = stage if fold is None else f"fold_{fold}"
    log_path = QUEUE_LOG_DIR / f"{name}.log"
    if log_path.exists():
        raise RuntimeError(f"E30 stage has a prior incomplete log; audit before retry: {log_path}")
    say("START", name)
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=6 * 3600)
    say("EXIT", name, result.returncode, log_path)
    if result.returncode != 0:
        raise RuntimeError(f"E30 {name} exited {result.returncode}; see {log_path}")
    if stage == "resource":
        report = verify_resource_complete(cfg)
        say("RESOURCE_GATE_PASS", "train_s", round(report["train_seconds"], 1),
            "infer_s", round(report["inference_seconds"], 1))
    elif stage == "fit-fold":
        verify_fold_complete(cfg, fold)


def verify_summary_complete(cfg: dict) -> None:
    analysis_dir = OUTPUT_DIR / "analysis"
    metrics = json.loads((analysis_dir / "metrics.json").read_text(encoding="utf-8"))
    expected = {
        "experiment": cfg["experiment"], "stage": "five_fold_oof_analysis",
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_lock_sha256": sha256(OUTPUT_DIR / "protocol_lock.json"),
        "rows": 2160, "prompts": 120,
        "holdout_evaluated": False, "fusion_weight_tuned": False,
    }
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Existing E30 summary differs from the locked protocol")
    for name in ("oof_predictions.csv", "fold_metrics.csv", "class_metrics.csv"):
        if not (analysis_dir / name).is_file():
            raise RuntimeError(f"Existing E30 summary is missing {name}")
    with (analysis_dir / "oof_predictions.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2160 or len({row.get("id") for row in rows}) != 2160:
        raise RuntimeError("Existing E30 summary OOF coverage is incomplete")


def run_summary(cfg: dict, env: dict[str, str]) -> None:
    analysis_dir = OUTPUT_DIR / "analysis"
    if analysis_dir.exists():
        verify_summary_complete(cfg)
        say("SKIP_COMPLETE", "summarize")
        return
    command = [sys.executable, "-u", str(ROOT / "run_e30_supervised_embedding8b.py"),
               "summarize", "--train", str(ROOT / "data/train.csv"),
               "--e0-dir", str(ROOT / "outputs/e0_tfidf_svm"),
               "--output-dir", str(OUTPUT_DIR)]
    log_path = QUEUE_LOG_DIR / "summarize.log"
    if log_path.exists():
        raise RuntimeError(f"E30 summarize has a prior incomplete log; audit before retry: {log_path}")
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"E30 summarize exited {result.returncode}; see {log_path}")
    verify_summary_complete(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Actually wait and launch the GPU stages")
    args = parser.parse_args()
    if not args.run:
        print("E30 queue is inert. Pass --run only after the E28 operator coordinates GPU release.")
        return
    acquire_lock()
    cfg = protocol_check()
    say("QUEUE_START", "pid", os.getpid(), "protocol", sha256(CONFIG_PATH))
    uuid = wait_for_ready(cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = uuid
    env["PYTHONPATH"] = str(ROOT / "outputs/e28_pydeps") + os.pathsep + str(ROOT)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    run_stage("resource", cfg, env)
    for fold in cfg["folds"]:
        run_stage("fit-fold", cfg, env, fold=fold)
    run_summary(cfg, env)
    say("E30_OOF_COMPLETE", OUTPUT_DIR / "analysis/metrics.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        say("QUEUE_ABORT", type(exc).__name__, str(exc))
        raise
