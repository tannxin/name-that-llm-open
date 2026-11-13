#!/usr/bin/env python3
"""Generate the offline pip-bootstrap v5, unscored E39 CUDA Notebook.

No competition data, model weights, adapters, scores or network requests are
used. The generator requires the final reviewed manifest SHA-256 and writes
only the versioned Notebook for the two-directory private Dataset layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = HERE / "e39_gpu_wheelhouse_spec.json"
MANIFEST = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118/e39_gpu_wheel_manifest.json"
LOCKED_MANIFEST_SHA256 = "a9a373ba8cf9d688b7730bdef811d4bd578416bda66bc1200f35af6accc881b8"
DESTINATION = HERE / "probe_e39_gpu_env_offline_unscored_v5.ipynb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_cell(source: str, name: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True), "id": name}


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["requirements"][0] != "torch==2.6.0+cu118":
        raise ValueError("GPU wheelhouse spec changed")
    if not MANIFEST.is_file() or sha256(MANIFEST) != LOCKED_MANIFEST_SHA256:
        raise ValueError("Final E39 split-Dataset manifest differs from the reviewed lock")
    expected_manifest_sha = LOCKED_MANIFEST_SHA256
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

input_root = pathlib.Path("/kaggle/input")
datasets_root = input_root / "datasets"
owner_root = datasets_root / "tanxin357"
dataset = owner_root / "name-that-llm-e39-gpu-wheelhouse-cp313-cu118"
if not input_root.is_dir() or input_root.is_symlink():
    raise RuntimeError("Expected real Kaggle input root")
for parent, child in ((input_root, datasets_root), (datasets_root, owner_root),
                      (owner_root, dataset)):
    if (not child.is_dir() or child.is_symlink() or
            child.resolve(strict=True).parent != parent.resolve(strict=True)):
        raise RuntimeError("Expected exact owner-scoped private E39 Dataset mount")
wheel_dirs = sorted(dataset.iterdir(), key=lambda path: path.name)
if (len(wheel_dirs) != 2 or any(not part.is_dir() or part.is_symlink() or
        part.resolve(strict=True).parent != dataset.resolve(strict=True)
        for part in wheel_dirs)):
    raise RuntimeError("Expected exactly two sibling ZIP extraction directories")
entries_by_dir = []
all_files = {}
for part in wheel_dirs:
    entries = list(part.iterdir())
    if len(entries) != 27:
        raise RuntimeError("Each E39 extracted ZIP must contain exactly 27 flat files")
    for path in entries:
        if (not path.is_file() or path.is_symlink() or
                path.resolve(strict=True).parent != part.resolve(strict=True)):
            raise RuntimeError("Unexpected nested or redirected E39 Dataset member")
        if path.name in all_files:
            raise RuntimeError("Duplicate E39 wheelhouse member across ZIP parts")
        all_files[path.name] = path
    entries_by_dir.append({path.name for path in entries})
manifest_parts = [part for part in wheel_dirs
                  if (part / "e39_gpu_wheel_manifest.json").is_file()]
if len(manifest_parts) != 1:
    raise RuntimeError("Expected exactly one E39 manifest, in part 1")
manifest_path = manifest_parts[0] / "e39_gpu_wheel_manifest.json"
if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("E39 GPU wheelhouse manifest SHA-256 differs from local lock")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest.get("schema_version") != 1 or
        manifest.get("spec_sha256") != EXPECTED_SPEC_SHA256 or
        manifest.get("selected_packages", {}).get("torch", {}).get("version") != "2.6.0+cu118"):
    raise RuntimeError("E39 GPU wheelhouse schema/spec/CUDA Torch lock differs")
expected = {item["file"] for item in manifest["wheels"]}
metadata_names = {"e39_gpu_wheel_manifest.json", "requirements-e39-gpu-locked.txt"}
if (len(expected) != 52 or len(manifest["wheels"]) != 52 or
        set(all_files) != expected | metadata_names or
        len(all_files) != 54 or
        {"e39_gpu_wheel_manifest.json", "requirements-e39-gpu-locked.txt"} -
        entries_by_dir[wheel_dirs.index(manifest_parts[0])] or
        len(entries_by_dir[wheel_dirs.index(manifest_parts[0])] & expected) != 25):
    raise RuntimeError("Combined E39 ZIP members differ from the locked 52-wheel set")
for item in manifest["wheels"]:
    filename = item["file"]
    if (pathlib.Path(filename).name != filename or chr(92) in filename or
            not filename.endswith(".whl")):
        raise RuntimeError("Unsafe E39 GPU wheel filename")
    path = all_files[filename]
    if path.is_symlink() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError("E39 GPU wheel hash or size differs: " + filename)
    if filename.startswith("torch-") and "+cu118" not in filename:
        raise RuntimeError("CPU or non-CUDA Torch wheel entered GPU environment")
locked = manifest_parts[0] / "requirements-e39-gpu-locked.txt"
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
import importlib.metadata
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time

environment = os.environ.copy()
for inherited in ("PIP_FIND_LINKS", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
                  "PIP_TRUSTED_HOST", "PIP_CONFIG_FILE"):
    environment.pop(inherited, None)
environment.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                   PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_XET="1", PIP_NO_CACHE_DIR="1", PIP_CONFIG_FILE=os.devnull)
try:
    base_pip_version = importlib.metadata.version("pip")
except importlib.metadata.PackageNotFoundError as error:
    raise RuntimeError("Base Kaggle Python lacks pip needed for offline --python bootstrap") from error
match = re.match(r"^(\\d+)\\.(\\d+)", base_pip_version)
if match is None or tuple(map(int, match.groups())) < (22, 3):
    raise RuntimeError("Base pip 22.3 or newer is required for offline --python bootstrap")
print({"stage": "base_pip_preflight", "version": base_pip_version})
venv = pathlib.Path("/kaggle/working/e39_gpu_cp313_cu118_venv_v5")
temporary = pathlib.Path("/kaggle/working/e39_gpu_pip_temp_v5")
if venv.exists() or venv.is_symlink() or temporary.exists() or temporary.is_symlink():
    raise RuntimeError("Use a fresh unscored Notebook run for the isolated offline environment")
temporary.mkdir()
environment["TMPDIR"] = str(temporary)
start = time.monotonic()
created = subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)],
                         env=environment, capture_output=True, text=True, timeout=120,
                         check=False)
if created.returncode:
    raise RuntimeError("Pip-free venv creation failed: " + created.stdout[-1000:] + created.stderr[-1000:])
gpu_python = venv / "bin/python"
if not gpu_python.is_file():
    raise RuntimeError("Pip-free E39 v5 interpreter is absent")
if gpu_python.is_symlink() and gpu_python.resolve(strict=True) != pathlib.Path(sys.executable).resolve(strict=True):
    raise RuntimeError("Pip-free E39 v5 interpreter symlink targets an unexpected executable")
config = venv / "pyvenv.cfg"
if (not config.is_file() or not any(line.strip().lower() ==
        "include-system-site-packages = false" for line in config.read_text().splitlines())):
    raise RuntimeError("E39 v5 environment permits system site packages")
probe_code = ("import sys,json,importlib.util;"
              "print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
              "'pip_visible':importlib.util.find_spec('pip') is not None}))")
probe = subprocess.run([str(gpu_python), "-c", probe_code], env=environment,
                       capture_output=True, text=True, timeout=30, check=False)
if probe.returncode:
    raise RuntimeError("Pip-free E39 v5 interpreter failed: " + probe.stderr[-1000:])
identity = json.loads(probe.stdout)
if (pathlib.Path(identity["prefix"]).resolve() != venv.resolve() or
        identity["base_prefix"] == identity["prefix"] or identity["pip_visible"]):
    raise RuntimeError("E39 v5 interpreter is not an isolated pip-free venv")
base_pip = [sys.executable, "-m", "pip", "--isolated", "--python", str(gpu_python)]
command = [*base_pip, "install", "--no-index", "--no-cache-dir",
           "--only-binary=:all:", "--require-hashes",
           "--find-links", str(wheel_dirs[0]), "--find-links", str(wheel_dirs[1]),
           "-r", str(locked)]
installed = subprocess.run(command, env=environment, capture_output=True, text=True,
                           timeout=1800, check=False)
if installed.returncode:
    raise RuntimeError("Offline GPU wheel install failed: " + installed.stdout[-1000:] + installed.stderr[-1000:])
checked = subprocess.run([*base_pip, "check"], env=environment,
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
                "# E39 offline GPU dependencies: unscored T4 x2 split-Dataset test (v5)\n",
                "\n",
                "Attach only the private, SHA-locked GPU wheelhouse Dataset containing both ZIP parts. "
                "Set T4 x2 and Internet off. This Notebook verifies all 54 extracted files across "
                "the two sibling directories, installs an isolated CPython 3.13 CUDA 11.8 "
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
