#!/usr/bin/env python3
"""One-shot E28 handoff: verified mirror download -> fixed one-fold resource run."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_e28_qwen8b_lora_feasibility import EXPECTED_SHA256, MODEL_REVISION, verify_model_dir


ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "outputs/e28_mirror_download"
DOWNLOAD_LOG = DOWNLOAD_DIR / "download.log"
MANIFEST = ROOT / "data/models/qwen3-8b-b968826d/e28_mirror_manifest.json"
MODEL_DIR = MANIFEST.parent
RESOURCE_DIR = ROOT / "outputs/e28_qwen8b_resource_mirror"
RESOURCE_LOG = DOWNLOAD_DIR / "resource.log"
TIMEOUT_SECONDS = 4 * 3600
POLL_SECONDS = 60


def say(*parts: object) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), *parts, flush=True)


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def download_complete() -> bool:
    if not MANIFEST.is_file() or not DOWNLOAD_LOG.is_file():
        return False
    return re.search(r"^COMPLETE\b", DOWNLOAD_LOG.read_text(encoding="utf-8", errors="replace"), re.MULTILINE) is not None


def verify_manifest_and_files() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("model_id") != "Qwen/Qwen3-8B":
        raise ValueError("Mirror manifest model ID differs from pinned E28 base")
    if manifest.get("hf_revision") != MODEL_REVISION:
        raise ValueError("Mirror manifest HF revision differs from pinned E28 revision")
    if manifest.get("all_sha256_verified") is not True:
        raise ValueError("Mirror manifest lacks full SHA256 verification flag")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 13:
        raise ValueError("Mirror manifest lacks the 13 locked runtime files")
    for name, digest in EXPECTED_SHA256.items():
        if files.get(name, {}).get("sha256") != digest:
            raise ValueError(f"Mirror manifest checksum differs: {name}")
    # This recomputes all required hashes from local bytes; the resource entry
    # repeats this check before any CUDA/model loading.
    verified_sizes = verify_model_dir(MODEL_DIR)
    for name, size in verified_sizes.items():
        if files[name]["size"] != size:
            raise ValueError(f"Mirror manifest size differs from local file: {name}")
    say("HASH_CHECK_PASS", len(verified_sizes), "required files")


def gpu2_idle() -> None:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [tuple(x.strip() for x in line.split(",")) for line in query.stdout.splitlines()]
    gpu2 = [row for row in rows if len(row) == 3 and row[0] == "2"]
    if len(gpu2) != 1:
        raise RuntimeError(f"Could not identify physical GPU2: {rows}")
    memory_mib, utilization = int(gpu2[0][1]), int(gpu2[0][2])
    if memory_mib > 256 or utilization != 0:
        raise RuntimeError(f"GPU2 is occupied: {memory_mib} MiB, {utilization}%")
    process_query = subprocess.run(
        ["nvidia-smi", "-i", "2", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    if re.search(r"\b\d+\b", process_query.stdout):
        raise RuntimeError(f"GPU2 has a compute process: {process_query.stdout.strip()}")
    say("GPU2_IDLE", memory_mib, "MiB", utilization, "%")


def main(download_pid: int) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    lock = DOWNLOAD_DIR / "watcher.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()) + "\n")
    say("WATCHER_START", "pid", os.getpid(), "download_pid", download_pid)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while not download_complete():
        if not pid_exists(download_pid):
            raise RuntimeError("Download process exited without COMPLETE and verified manifest")
        if time.monotonic() >= deadline:
            raise TimeoutError("Download did not complete within the one-shot watcher timeout")
        time.sleep(POLL_SECONDS)
    say("DOWNLOAD_COMPLETE_SIGNAL")
    verify_manifest_and_files()
    if RESOURCE_DIR.exists() and any(RESOURCE_DIR.iterdir()):
        raise RuntimeError("Resource output directory is already populated")
    gpu2_idle()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTHONPATH"] = str(ROOT / "outputs/e28_pydeps") + os.pathsep + str(ROOT)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        sys.executable, "-u", str(ROOT / "run_e28_qwen8b_lora_feasibility.py"), "resource",
        "--model-dir", str(MODEL_DIR),
        "--train", str(ROOT / "data/train.csv"),
        "--e0-dir", str(ROOT / "outputs/e0_tfidf_svm"),
        "--output-dir", str(RESOURCE_DIR),
    ]
    say("START_RESOURCE", "fold1", "one_epoch", "4000_inference", "gpu2")
    with RESOURCE_LOG.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=4 * 3600 + 1800)
    say("RESOURCE_EXIT", result.returncode, "log", RESOURCE_LOG)
    if result.returncode != 0:
        raise RuntimeError(f"E28 resource script failed with exit code {result.returncode}")
    report_path = RESOURCE_DIR / "resource_report.json"
    if not report_path.is_file():
        raise RuntimeError("E28 resource script exited cleanly without its resource report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("validation_accuracy_computed") is not False or report.get("validation_scores_saved") is not False:
        raise RuntimeError("Resource report violated no-validation-score protocol")
    say("RESOURCE_COMPLETE", "gate_passed", report.get("gate_passed"), "train_s", round(report["training_seconds"], 1), "infer_s", round(report["inference_seconds"], 1))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--download-pid", required=True, type=int)
        options = parser.parse_args()
        if options.download_pid <= 0:
            parser.error("--download-pid must be positive")
        main(options.download_pid)
    except Exception as exc:
        say("WATCHER_ABORT", type(exc).__name__, str(exc))
        raise
