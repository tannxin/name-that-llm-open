#!/usr/bin/env python3
"""Generate the private, unscored E39 offline CUDA-dependency test Notebook.

No competition data, model weights, adapters, scores or network requests are
used. Run this builder again after the GPU wheelhouse manifest is complete to
embed its exact SHA-256; an incomplete build yields a fail-closed Notebook.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = HERE / "e39_gpu_wheelhouse_spec.json"
MANIFEST = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118/e39_gpu_wheel_manifest.json"
DESTINATION = HERE / "probe_e39_gpu_env_offline_unscored.ipynb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_cell(source: str, name: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True), "id": name}


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["requirements"][0] != "torch==2.6.0+cu118":
        raise ValueError("GPU wheelhouse spec changed")
    expected_manifest_sha = sha256(MANIFEST) if MANIFEST.is_file() else "REPLACE_AFTER_WHEELHOUSE_BUILD"
    preflight = '''import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

EXPECTED_SPEC_SHA256 = "__E39_SPEC_DIGEST__"
EXPECTED_MANIFEST_SHA256 = "__E39_MANIFEST_DIGEST__"
if EXPECTED_MANIFEST_SHA256.startswith("REPLACE_"):
    raise RuntimeError("Build and verify the offline wheelhouse before this Notebook can run")
if sys.version_info[:2] != (3, 13) or sys.implementation.cache_tag != "cpython-313":
    raise RuntimeError("This wheelhouse is built only for regular CPython 3.13 (cp313)")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

matches = list(pathlib.Path("/kaggle/input").rglob("e39_gpu_wheel_manifest.json"))
if len(matches) != 1:
    raise RuntimeError("Attach exactly one private E39 GPU wheelhouse Dataset")
manifest_path = matches[0]
wheelhouse = manifest_path.parent.resolve(strict=True)
if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("E39 GPU wheelhouse manifest SHA-256 differs from local lock")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest.get("schema_version") != 1 or
        manifest.get("spec_sha256") != EXPECTED_SPEC_SHA256 or
        manifest.get("selected_packages", {}).get("torch", {}).get("version") != "2.6.0+cu118"):
    raise RuntimeError("E39 GPU wheelhouse schema/spec/CUDA Torch lock differs")
expected = {item["file"] for item in manifest["wheels"]}
actual = {path.name for path in wheelhouse.glob("*.whl")}
if len(expected) != len(manifest["wheels"]) or actual != expected:
    raise RuntimeError("E39 GPU wheel list differs from the local lock")
for item in manifest["wheels"]:
    filename = item["file"]
    if pathlib.Path(filename).name != filename or not filename.endswith(".whl"):
        raise RuntimeError("Unsafe E39 GPU wheel filename")
    path = wheelhouse / filename
    if path.is_symlink() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError("E39 GPU wheel hash or size differs: " + filename)
    if filename.startswith("torch-") and "+cu118" not in filename:
        raise RuntimeError("CPU or non-CUDA Torch wheel entered GPU environment")
locked = wheelhouse / "requirements-e39-gpu-locked.txt"
if sha256(locked) != manifest["locked_requirements_sha256"]:
    raise RuntimeError("E39 GPU hash-locked requirement file changed")

work = pathlib.Path("/kaggle/working")
free_gib = shutil.disk_usage(work).free / 2**30
if free_gib < 10:
    raise RuntimeError("At least 10 GiB writable space required for isolated offline install")
print(json.dumps({"stage": "verified_wheels", "count": len(expected),
                  "wheel_gib": round(sum(item["bytes"] for item in manifest["wheels"]) / 2**30, 2),
                  "working_free_gib": round(free_gib, 2)}))
'''.replace("__E39_SPEC_DIGEST__", sha256(SPEC)).replace(
        "__E39_MANIFEST_DIGEST__", expected_manifest_sha)
    install = '''import os
import pathlib
import shutil
import subprocess
import sys
import time

environment = os.environ.copy()
environment.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                   PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_XET="1", PIP_NO_CACHE_DIR="1")
venv = pathlib.Path("/kaggle/working/e39_gpu_cp313_cu118_venv")
temporary = pathlib.Path("/kaggle/working/e39_gpu_pip_temp")
if venv.exists() or temporary.exists():
    raise RuntimeError("Use a fresh unscored Notebook run for the isolated offline environment")
temporary.mkdir()
environment["TMPDIR"] = str(temporary)
start = time.monotonic()
subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=120)
gpu_python = venv / "bin/python"
command = [str(gpu_python), "-m", "pip", "install", "--no-index", "--no-cache-dir",
           "--only-binary=:all:", "--require-hashes", "--find-links", str(wheelhouse),
           "-r", str(locked)]
installed = subprocess.run(command, env=environment, capture_output=True, text=True,
                           timeout=1800, check=False)
if installed.returncode:
    raise RuntimeError("Offline GPU wheel install failed: " + installed.stdout[-1000:] + installed.stderr[-1000:])
checked = subprocess.run([str(gpu_python), "-m", "pip", "check"], env=environment,
                         capture_output=True, text=True, timeout=60, check=False)
if checked.returncode:
    raise RuntimeError("Offline GPU dependency check failed: " + checked.stdout[-1000:] + checked.stderr[-1000:])
print({"stage": "offline_install", "seconds": round(time.monotonic() - start, 1),
       "working_free_gib": round(shutil.disk_usage("/kaggle/working").free / 2**30, 2)})
'''
    smoke = '''import json
import os
import subprocess

smoke_code = r"""
import importlib.metadata
import json
import torch
import bitsandbytes as bnb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import BitsAndBytesConfig, Qwen3ForSequenceClassification
from peft import PeftModel
import accelerate

expected = {"torch": "2.6.0+cu118", "transformers": "4.51.3", "peft": "0.15.2",
            "accelerate": "1.6.0", "bitsandbytes": "0.45.5"}
versions = {name: importlib.metadata.version(name) for name in expected}
if versions != expected or torch.version.cuda != "11.8":
    raise RuntimeError("Pinned E39 GPU versions/CUDA differ")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("Expected two CUDA devices")
buffer = pa.BufferOutputStream()
original = pa.table({"probe": pa.array([1], type=pa.int8())})
pq.write_table(original, buffer)
if not original.equals(pq.read_table(pa.BufferReader(buffer.getvalue()))):
    raise RuntimeError("Pinned GPU environment pyarrow Parquet roundtrip failed")
devices = []
for index in range(2):
    device = torch.device(f"cuda:{index}")
    name = torch.cuda.get_device_name(index)
    if "T4" not in name:
        raise RuntimeError("Expected Kaggle T4 x2")
    with torch.inference_mode():
        layer = bnb.nn.Linear4bit(128, 128, bias=False, compute_dtype=torch.float16,
                                  compress_statistics=True, quant_type="nf4").to(device)
        result = layer(torch.ones((1, 128), dtype=torch.float16, device=device))
        torch.cuda.synchronize(index)
        if result.shape != (1, 128) or not bool(torch.isfinite(result).all()):
            raise RuntimeError("Tiny NF4 forward failed")
    free, total = torch.cuda.mem_get_info(index)
    devices.append({"name": name, "total_mib": total // 2**20, "free_mib": free // 2**20,
                    "nf4_tiny_forward": "pass"})
print(json.dumps({"stage": "tiny_nf4_import_smoke", "versions": versions,
                  "torch_cuda": torch.version.cuda, "devices": devices}))
"""
result = subprocess.run([str(gpu_python), "-c", smoke_code], env=environment,
                        capture_output=True, text=True, timeout=180, check=False)
if result.returncode:
    raise RuntimeError("Pinned E39 GPU import/NF4 smoke failed: " + result.stdout[-1000:] + result.stderr[-1000:])
print(result.stdout[-2000:])
print("E39_GPU_ENV_UNSCORED_PASS: no model/data loaded; no score or submission")
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# E39 offline GPU dependencies: unscored T4 x2 test\n",
                "\n",
                "Attach only the private, SHA-locked GPU wheelhouse. Set T4 x2 and Internet off. "
                "This Notebook verifies wheel hashes, installs an isolated CPython 3.13 CUDA 11.8 "
                "environment, imports the locked E39 stack, and runs one tiny NF4 forward per card. "
                "It does not load competition data, model weights, or an adapter, and writes no submission. "
                "Do not click Submit. A pass does not establish E39 deployment feasibility.\n"]},
            code_cell(preflight, "e39-gpu-wheel-preflight"),
            code_cell(install, "e39-gpu-offline-install"),
            code_cell(smoke, "e39-gpu-tiny-nf4"),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(DESTINATION), "manifest_sha256": expected_manifest_sha}))


if __name__ == "__main__":
    main()
