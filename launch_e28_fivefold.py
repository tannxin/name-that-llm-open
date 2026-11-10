#!/usr/bin/env python3
"""Run the locked E28 GPU2 folds sequentially and release GPU2 after exit.

Launch from the project root with ``nohup python -u launch_e28_fivefold.py ... &``.
This runner never downloads weights, changes hyperparameters, or evaluates the
held-out prompts. It can resume only at a fully completed fold boundary.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_e0 import sha256


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_e28_qwen8b_fivefold.py"
OUTPUT = ROOT / "outputs/e28_qwen8b_fivefold"
MODEL = ROOT / "data/models/qwen3-8b-b968826d"
RESOURCE = ROOT / "outputs/e28_qwen8b_resource_mirror/resource_report.json"
RELEASE = OUTPUT / "gpu2_released.ok"


def log(*items) -> None:
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), *items, flush=True)


def gpu2_idle() -> str:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
        timeout=20)
    rows = [tuple(part.strip() for part in line.split(","))
            for line in query.stdout.splitlines()]
    matching = [row for row in rows if len(row) == 4 and row[0] == "2"]
    if len(matching) != 1:
        raise RuntimeError("Physical GPU2 cannot be identified")
    _, uuid, used, utilization = matching[0]
    if not uuid.startswith("GPU-") or int(used) > 256 or int(utilization) != 0:
        raise RuntimeError(f"Physical GPU2 is busy: {used} MiB, {utilization}%")
    processes = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
         "--format=csv,noheader"], check=True, capture_output=True, text=True,
        timeout=20)
    if re.search(r"\b\d+\b", processes.stdout):
        raise RuntimeError("Physical GPU2 still has a compute process")
    return uuid


def wait_gpu2_idle(max_seconds: float = 180) -> str:
    deadline = time.monotonic() + max_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            return gpu2_idle()
        except RuntimeError as error:
            last_error = error
            time.sleep(5)
    raise RuntimeError(f"GPU2 did not become idle after process exit: {last_error}")


def run(command: list[str], env: dict[str, str], log_path: Path, timeout: float) -> None:
    log("START", *command, "log", log_path)
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    log("EXIT", result.returncode, "log", log_path)
    if result.returncode != 0:
        raise RuntimeError(f"E28 command failed with exit code {result.returncode}: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-script-sha256", required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if RELEASE.exists():
        raise RuntimeError("E28 release marker already exists; do not restart blindly")
    lock_path = OUTPUT / "runner.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        success = False
        try:
            if sha256(SCRIPT) != args.expected_script_sha256:
                raise RuntimeError("E28 script differs from expected deployment SHA-256")
            report = json.loads(RESOURCE.read_text(encoding="utf-8"))
            if report.get("gate_passed") is not True:
                raise RuntimeError("E28 resource gate is not passed")
            uuid = gpu2_idle()
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": uuid,
                "PYTHONPATH": str(ROOT / "outputs/e28_pydeps") + os.pathsep + str(ROOT),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            })
            base = [sys.executable, "-u", str(SCRIPT)]
            preflight_log = OUTPUT / "preflight.log"
            if not (OUTPUT / "protocol_lock.json").exists():
                run(base + ["preflight"], env, preflight_log, 300)
            else:
                # Preflight checks the full immutable lock without rewriting it.
                if preflight_log.exists():
                    preflight_log = OUTPUT / "preflight_resume.log"
                run(base + ["preflight"], env, preflight_log, 300)
            for fold in range(1, 6):
                folder = OUTPUT / f"fold_{fold}"
                if folder.exists():
                    if not (folder / "run.json").is_file():
                        raise RuntimeError(f"E28 fold {fold} has partial output; manual audit needed")
                    run_meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
                    if (run_meta.get("fold") != fold or run_meta.get("training_epochs_completed") != 3
                            or run_meta.get("oof_predictions_sha256") != sha256(folder / "oof_predictions.csv")):
                        raise RuntimeError(f"E28 fold {fold} resume metadata differs")
                    log("SKIP_COMPLETE_FOLD", fold)
                    continue
                wait_gpu2_idle()
                run(base + ["fit-fold", "--fold", str(fold), "--model-dir", str(MODEL)],
                    env, OUTPUT / f"fold_{fold}.log", 4 * 3600)
            wait_gpu2_idle()
            run(base + ["summarize"], env, OUTPUT / "summarize.log", 1200)
            success = True
            log("E28_FIVEFOLD_COMPLETE", "analysis", OUTPUT / "analysis/metrics.json")
        finally:
            # A release marker denotes actual GPU availability, even if E28 failed.
            # If another process occupies GPU2, the marker is deliberately absent.
            uuid = wait_gpu2_idle()
            (OUTPUT / "gpu2_release_status.json").write_text(json.dumps({
                "experiment": "e28_qwen3_8b_official_only_fivefold",
                "gpu2_uuid": uuid,
                "gpu2_idle_at_release": True,
                "e28_success": success,
                "released_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2) + "\n", encoding="utf-8")
            # The independent E30 queue requires this exact one-line marker.
            RELEASE.write_text("E28_GPU2_RELEASED_FOR_E30\n", encoding="utf-8")
            log("GPU2_RELEASED", "success", success, "marker", RELEASE)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("E28_RUNNER_ABORT", type(exc).__name__, str(exc))
        raise
