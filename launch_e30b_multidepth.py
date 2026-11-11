#!/usr/bin/env python3
"""Run the locked E30-B resource gate and five folds on physical GPU2.

This is a one-shot operator launcher. It requires the existing CPU preflight
lock and never creates or rewrites that lock. Completed stages may be resumed
only after their metadata and artifacts pass the checks below; partial stages
require a manual audit. Fold scores are not read or reported between stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import run_e30b_multidepth as b
import run_e30_supervised_embedding8b as a
from run_e0 import LABELS, sha256


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_e30b_multidepth.py"
CONFIG = ROOT / "configs/e30b_qwen3_embedding8b_multidepth.json"
MODEL = ROOT / "data/models/qwen3-embedding-8b-1d8ad4c"
OUTPUT = ROOT / "outputs/e30b_qwen3_embedding8b_v1"
A_OUTPUT = ROOT / "outputs/e30_qwen3_embedding8b_v3"
QUEUE = ROOT / "outputs/e30b_gpu_queue"
LOCK = OUTPUT / "protocol_lock.json"
RESOURCE = OUTPUT / "resource_report.json"
FINAL_STATUS = QUEUE / "runner_status.json"
FINISHED = QUEUE / "runner_finished.ok"
RELEASE = QUEUE / "gpu2_released.ok"

# These are the hashes recorded before E30-B validation in its preregistration.
EXPECTED_SCRIPT_SHA256 = "d660aa5e70cb25066cdecdd1e9425e8328cbdc68b655753e6f75443433f8895b"
EXPECTED_CONFIG_SHA256 = "b46f82add551e7440c4fc2dd4ae9fccfec5aa9ea627df0fc2fbe68d7e4ce5d09"
RESOURCE_GATE_KEYS = {
    "train_peak_under_limit", "inference_peak_under_limit",
    "one_epoch_under_limit", "projected_three_epochs_under_limit",
    "four_thousand_inference_under_limit", "optimizer_updates_under_skip_limit",
    "finite_four_thousand_logits",
}


def say(*parts: object) -> None:
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), *parts, flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def locked_args() -> SimpleNamespace:
    return SimpleNamespace(
        train=ROOT / "data/train.csv",
        e0_dir=ROOT / "outputs/e0_tfidf_svm",
        a_output_dir=A_OUTPUT,
        output_dir=OUTPUT,
    )


def check_protocol() -> dict:
    if sha256(SCRIPT) != EXPECTED_SCRIPT_SHA256:
        raise RuntimeError("E30-B script differs from the preregistered SHA-256")
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("E30-B config differs from the preregistered SHA-256")
    cfg = b.config()  # Also pins the reused E30-A script and config.
    lock = b.lock_input(locked_args(), cfg)
    expected = {
        "stage": "protocol_lock_before_e30b_validation",
        "readout_indices": [12, 24, 36],
        "readout_cpu_contract_passed": True,
        "development_rows": 2160,
        "development_prompts": 120,
        "fold_rows": {str(fold): 432 for fold in range(1, 6)},
        "validation_accuracy_computed": False,
        "b_model_loaded": False,
        "gpu_used": False,
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E30-B CPU preflight lock is missing or differs")
    b.cpu_readout_contract(cfg)
    return cfg


def gpu2_state() -> tuple[str, bool]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [row for row in csv.reader(query.stdout.splitlines()) if row]
    matching = [[part.strip() for part in row] for row in rows
                if len(row) == 5 and row[0].strip() == "2"]
    if len(matching) != 1 or "RTX 3090" not in matching[0][2]:
        raise RuntimeError("Cannot identify physical RTX 3090 GPU2")
    _, uuid, _, used, utilization = matching[0]
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
        raise RuntimeError("GPU2 UUID format changed")
    processes = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    has_compute_pid = bool(re.search(r"(?m)^\s*\d+\s*$", processes.stdout))
    return uuid, int(used) <= 256 and int(utilization) == 0 and not has_compute_pid


def wait_gpu2_idle(expected_uuid: str | None, seconds: float) -> str:
    deadline = time.monotonic() + seconds
    last_state = "unavailable"
    while True:
        uuid, idle = gpu2_state()
        if expected_uuid is not None and uuid != expected_uuid:
            raise RuntimeError("Physical GPU2 UUID changed while E30-B was queued")
        if idle:
            return uuid
        last_state = f"{uuid} busy"
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU2 did not become idle: {last_state}")
        time.sleep(min(5, max(0.1, deadline - time.monotonic())))


def verify_resource(cfg: dict) -> dict:
    report = b.checked_resource(locked_args(), cfg)
    if (report.get("stage") != "one_fold_resource_only"
            or report.get("script_sha256") != EXPECTED_SCRIPT_SHA256
            or report.get("model_file_sha256") != a.EXPECTED_SHA256
            or report.get("fold_training_side") != 1
            or report.get("training_rows") != 1728
            or report.get("training_epochs") != 1
            or report.get("unlabeled_inference_rows") != 4000
            or report.get("validation_accuracy_computed") is not False
            or report.get("gate_passed") is not True
            or set(report.get("gate", {})) != RESOURCE_GATE_KEYS
            or any(value is not True for value in report["gate"].values())):
        raise RuntimeError("E30-B resource report differs from the locked gate")
    if (report.get("train_peak_reserved_bytes", float("inf")) >= 22 * 2**30
            or report.get("inference_peak_reserved_bytes", float("inf")) >= 22 * 2**30):
        raise RuntimeError("E30-B resource memory exceeded 22 GiB")
    resource_dir = OUTPUT / "resource_one_epoch"
    for relative in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                     "head.safetensors", "tokenizer/tokenizer.json"):
        if not (resource_dir / relative).is_file():
            raise RuntimeError(f"E30-B resource artifact missing: {relative}")
    return report


def verify_fold(cfg: dict, fold: int) -> None:
    folder = OUTPUT / f"fold_{fold}"
    prediction_file = folder / "oof_predictions.csv"
    run = read_json(folder / "run.json")
    expected = {
        "experiment": cfg["experiment"], "fold": fold,
        "config_sha256": sha256(CONFIG),
        "script_sha256": EXPECTED_SCRIPT_SHA256,
        "protocol_lock_sha256": sha256(LOCK),
        "resource_report_sha256": sha256(RESOURCE),
        "model_revision": cfg["model_revision"],
        "model_file_sha256": a.EXPECTED_SHA256,
        "epochs": 3, "train_rows": 1728, "validation_rows": 432,
        "oof_predictions_sha256": sha256(prediction_file),
        "no_epoch_or_threshold_selection": True, "holdout_evaluated": False,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"E30-B fold {fold} metadata differs from the lock")
    if (run.get("train_peak_reserved_bytes", float("inf")) >= 22 * 2**30
            or run.get("completed_optimizer_updates", 0)
            < 0.95 * run.get("planned_optimizer_updates", 1)):
        raise RuntimeError(f"E30-B fold {fold} exceeded resource/update limits")
    for relative in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                     "head.safetensors", "tokenizer/tokenizer.json"):
        if not (folder / relative).is_file():
            raise RuntimeError(f"E30-B fold {fold} artifact missing: {relative}")
    # Identity and coverage only: no logits or validation metrics are inspected.
    with prediction_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "prompt_id", "true_model", "fold", "pred_model"}
        required |= {f"score_{label}" for label in LABELS}
        if set(reader.fieldnames or ()) != required:
            raise RuntimeError(f"E30-B fold {fold} OOF columns differ")
        count, ids, prompts = 0, set(), set()
        for row in reader:
            count += 1
            ids.add(row["id"])
            prompts.add(row["prompt_id"])
            if row["fold"] != str(fold) or row["true_model"] not in LABELS:
                raise RuntimeError(f"E30-B fold {fold} OOF identity differs")
    if count != 432 or len(ids) != 432 or len(prompts) != 24:
        raise RuntimeError(f"E30-B fold {fold} OOF coverage incomplete")


def verify_summary(cfg: dict) -> None:
    analysis = OUTPUT / "analysis"
    metrics = read_json(analysis / "metrics.json")
    expected = {
        "experiment": cfg["experiment"], "stage": "fivefold_oof_analysis",
        "config_sha256": sha256(CONFIG),
        "script_sha256": EXPECTED_SCRIPT_SHA256,
        "protocol_lock_sha256": sha256(LOCK),
        "rows": 2160, "prompts": 120,
        "holdout_evaluated": False, "fusion_weight_tuned": False,
        "deployment_and_kaggle_proven": False,
    }
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E30-B summary metadata differs from the lock")
    for name in ("oof_predictions.csv", "fold_metrics.csv", "class_metrics.csv"):
        if not (analysis / name).is_file():
            raise RuntimeError(f"E30-B summary artifact missing: {name}")
    # The summarizer performs row, prompt, truth, fold, finite-logit, and
    # comparator-hash audits. This launcher deliberately does not print scores.


def stage_command(stage: str, fold: int | None = None) -> list[str]:
    command = [
        sys.executable, "-u", str(SCRIPT), stage,
        "--train", str(ROOT / "data/train.csv"),
        "--e0-dir", str(ROOT / "outputs/e0_tfidf_svm"),
        "--a-output-dir", str(A_OUTPUT),
        "--output-dir", str(OUTPUT),
    ]
    if stage in {"resource", "fit-fold"}:
        command.extend(["--model-dir", str(MODEL)])
    if fold is not None:
        command.extend(["--fold", str(fold)])
    return command


def run_stage(stage: str, cfg: dict, gpu_env: dict[str, str],
              uuid: str, *, fold: int | None = None) -> None:
    name = "resource" if stage == "resource" else f"fold_{fold}"
    if stage == "resource" and RESOURCE.exists():
        verify_resource(cfg)
        say("SKIP_VERIFIED", name)
        return
    if stage == "fit-fold" and (OUTPUT / name).exists():
        verify_fold(cfg, fold)
        say("SKIP_VERIFIED", name)
        return
    if stage == "resource" and (OUTPUT / "resource_one_epoch").exists():
        raise RuntimeError("E30-B resource directory is partial; manual audit required")
    log_path = QUEUE / f"{name}.log"
    if log_path.exists():
        raise RuntimeError(f"E30-B {name} has a prior log without complete output")
    wait_gpu2_idle(uuid, 30 * 60)
    say("START", name)
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(stage_command(stage, fold), cwd=ROOT, env=gpu_env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=6 * 3600)
    say("EXIT", name, result.returncode, log_path)
    if result.returncode != 0:
        raise RuntimeError(f"E30-B {name} exited {result.returncode}; see {log_path}")
    if stage == "resource":
        verify_resource(cfg)
        say("RESOURCE_GATE_PASS")
    else:
        verify_fold(cfg, fold)
        say("FOLD_COMPLETE", fold, "identity and resource checks only")


def run_summary(cfg: dict, cpu_env: dict[str, str]) -> None:
    analysis = OUTPUT / "analysis"
    if analysis.exists():
        verify_summary(cfg)
        say("SKIP_VERIFIED", "summarize")
        return
    log_path = QUEUE / "summarize.log"
    if log_path.exists():
        raise RuntimeError("E30-B summarize has a prior log without complete output")
    say("START", "summarize CPU")
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(stage_command("summarize"), cwd=ROOT, env=cpu_env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=3600)
    say("EXIT", "summarize", result.returncode, log_path)
    if result.returncode != 0:
        raise RuntimeError(f"E30-B summarize exited {result.returncode}; see {log_path}")
    verify_summary(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Actually launch the locked GPU stages")
    args = parser.parse_args()
    if not args.run:
        print("E30-B launcher is inert; pass --run after reviewing the CPU lock.")
        return
    import fcntl  # The active launcher runs on the Linux Pod.

    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "runner.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if FINAL_STATUS.exists() or FINISHED.exists() or RELEASE.exists():
            raise RuntimeError("E30-B runner already finalized; manual audit required")
        bound_uuid: str | None = None
        run_complete = False
        failure: BaseException | None = None
        try:
            cfg = check_protocol()
            say("PROTOCOL_LOCK_PASS", sha256(LOCK))
            a.verify_model_dir(MODEL, cfg)
            say("PINNED_MODEL_HASH_PASS")
            bound_uuid = wait_gpu2_idle(None, 48 * 3600)
            say("GPU2_IDLE", bound_uuid)
            gpu_env = os.environ.copy()
            gpu_env.update({
                "CUDA_VISIBLE_DEVICES": bound_uuid,
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": os.pathsep.join(filter(None, (
                    str(ROOT / "outputs/e28_pydeps"), str(ROOT),
                    gpu_env.get("PYTHONPATH", "")))),
            })
            run_stage("resource", cfg, gpu_env, bound_uuid)
            for fold in cfg["folds"]:
                run_stage("fit-fold", cfg, gpu_env, bound_uuid, fold=fold)
            wait_gpu2_idle(bound_uuid, 30 * 60)
            cpu_env = gpu_env.copy()
            cpu_env["CUDA_VISIBLE_DEVICES"] = ""
            run_summary(cfg, cpu_env)
            run_complete = True
            say("E30B_FIVEFOLD_COMPLETE", "analysis present; independent audit pending")
        except BaseException as exc:
            failure = exc
            say("E30B_RUNNER_ABORT", type(exc).__name__, str(exc))
            raise
        finally:
            idle_at_release = False
            release_error = None
            if bound_uuid is not None:
                try:
                    wait_gpu2_idle(bound_uuid, 180)
                    idle_at_release = True
                except Exception as exc:
                    release_error = f"{type(exc).__name__}: {exc}"
            success = run_complete and idle_at_release and failure is None
            write_new_json(FINAL_STATUS, {
                "experiment": "e30_b_qwen3_embedding8b_multidepth_official_only",
                "runner_success": success,
                "fivefold_and_summary_complete": run_complete,
                "gpu2_uuid": bound_uuid,
                "gpu2_idle_at_release": idle_at_release,
                "error": None if failure is None else f"{type(failure).__name__}: {failure}",
                "release_error": release_error,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            FINISHED.write_text(
                "E30B_RUNNER_SUCCESS\n" if success else "E30B_RUNNER_FAILURE\n",
                encoding="utf-8")
            if idle_at_release:
                RELEASE.write_text(
                    "E30B_GPU2_RELEASED_SUCCESS\n" if success
                    else "E30B_GPU2_RELEASED_FAILURE\n", encoding="utf-8")
            say("RUNNER_FINISHED", "success", success,
                "gpu2_idle", idle_at_release, "status", FINAL_STATUS)
            if run_complete and not idle_at_release and failure is None:
                raise RuntimeError("E30-B analysis completed, but GPU2 release was not verified")


if __name__ == "__main__":
    main()
