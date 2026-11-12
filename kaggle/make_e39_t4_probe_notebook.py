#!/usr/bin/env python3
"""Build an unscored Kaggle T4 x2 deployment probe from the audited E24 setup.

This is deliberately a proxy Notebook. It writes a nested Parquet file with
synthetic IDs, and its version must not be submitted to the competition.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
E24_NOTEBOOK = HERE / "submit_name_that_llm_e24.ipynb"
DESTINATION = HERE / "probe_name_that_llm_e39_t4_unscored.ipynb"
FILES = (
    "submit_e24_external_e0c.py",
    "e39_probe_common.py",
    "score_e24_fixed_probe.py",
    "probe_e39_gpu.py",
    "e39_probe_asset_manifest.json",
)


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def cell(source: str, name: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(source), "id": name}


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    source = json.loads(E24_NOTEBOOK.read_text(encoding="utf-8"))
    notebook = copy.deepcopy(source)
    notebook["cells"] = notebook["cells"][:7]
    notebook["cells"][0]["source"] = lines(
        "# E39 unscored T4 deployment probe\n\n"
        "This private Notebook checks whether the fixed E24 + 1×E39 pipeline can "
        "load and process 4,000 **unlabelled training-side** texts under Internet "
        "off and Kaggle T4×2. It uses an existing fold-1 E39 adapter, not a full "
        "submission adapter. It writes synthetic IDs to a nested Parquet file "
        "for schema testing. Do not click **Submit** on this version. The original "
        "E24 Notebook remains the submitted baseline.\n"
    )
    notebook["cells"][1]["source"] += lines(
        "\nimport sys\nimport time\n"
        "PROBE_STARTED = time.monotonic()\n"
        "PROBE_ROOT = Path('/kaggle/working/e39_unscored_probe')\n"
        "OUTPUT_PATH = PROBE_ROOT / 'submission.parquet'  # Never the competition output path.\n"
        "BASE_DIR = None  # Pinned Qwen3-8B-Base Dataset directory; auto-detect if unique.\n"
        "ADAPTER_DIR = None  # Private fold-1 E39 adapter directory; auto-detect if unique.\n"
        "SMOKE_REFERENCE_PATH = None  # Private, train-side 3090 reference JSON.\n"
        "GPU_PYTHON = Path(sys.executable)  # Set to verified offline GPU venv if needed.\n"
        "DEVICE_MODE = 'single'  # Fixed first test; use 'dual' only after single-T4 OOM.\n"
        "TIME_LIMIT_SECONDS = 8 * 3600  # One hour below the competition's nine-hour cap.\n"
    )

    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in FILES}
    hashes = {name: hashlib.sha256(value.encode("utf-8")).hexdigest()
              for name, value in sources.items()}
    stage = '''import hashlib
import os
import shutil
import subprocess

WORK_SOURCE = Path('/kaggle/working/name_that_llm_source_e39_probe')
shutil.copytree(SOURCE_DIR, WORK_SOURCE, dirs_exist_ok=True)
shutil.copy2(MODERN_SOURCE_PATH, WORK_SOURCE / 'fit_modernbert_full.py')
for name in ('fit_e24_full_e0c.py', 'run_e24_prvsiyan_external_ablation.py'):
    shutil.copy2(E24_SOURCE_DIR / name, WORK_SOURCE / name)
WORK_KAGGLE = WORK_SOURCE / 'kaggle'
WORK_KAGGLE.mkdir(parents=True, exist_ok=True)
SOURCE_TEXT = SOURCES
SOURCE_SHA256 = HASHES
for name, value in SOURCE_TEXT.items():
    target = WORK_KAGGLE / name
    target.write_text(value, encoding='utf-8')
    if hashlib.sha256(target.read_bytes()).hexdigest() != SOURCE_SHA256[name]:
        raise RuntimeError(f'E39 probe source hash mismatch: {name}')
WORK_MANIFEST = WORK_KAGGLE / 'e39_probe_asset_manifest.json'
GPU_SCRIPT = WORK_KAGGLE / 'probe_e39_gpu.py'
CPU_SCRIPT = WORK_KAGGLE / 'score_e24_fixed_probe.py'
if DEVICE_MODE not in ('single', 'dual'):
    raise ValueError('DEVICE_MODE must be single or dual')
PROBE_ROOT.mkdir(parents=True, exist_ok=False)
print('E39 probe source and asset manifest hashes verified')
'''.replace("SOURCES", repr(sources)).replace("HASHES", repr(hashes))
    notebook["cells"].append(cell(stage, "e39-stage"))

    locate = '''import json

asset_manifest = json.loads(WORK_MANIFEST.read_text(encoding='utf-8'))
def one_match(paths, kind):
    candidates = sorted(set(path.resolve(strict=True) for path in paths))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one {kind} directory, found {len(candidates)}; set it explicitly')
    return candidates[0]

if BASE_DIR is None:
    expected = asset_manifest['model_files']['config.json']['sha256']
    BASE_DIR = one_match((p.parent for p in INPUT_ROOT.rglob('config.json')
                          if sha256_file(p) == expected), 'pinned Qwen3-8B-Base')
else:
    BASE_DIR = Path(BASE_DIR).resolve(strict=True)
if ADAPTER_DIR is None:
    expected = asset_manifest['adapter_files']['adapter_model.safetensors']
    ADAPTER_DIR = one_match((p.parent for p in INPUT_ROOT.rglob('adapter_model.safetensors')
                             if sha256_file(p) == expected), 'private E39 fold-1 adapter')
else:
    ADAPTER_DIR = Path(ADAPTER_DIR).resolve(strict=True)
if SMOKE_REFERENCE_PATH is None:
    paths = list(INPUT_ROOT.rglob('e39_fold1_smoke_reference.json'))
    if len(paths) != 1:
        raise RuntimeError('Attach exactly one private train-side smoke reference')
    SMOKE_REFERENCE_PATH = paths[0]
SMOKE_REFERENCE_PATH = Path(SMOKE_REFERENCE_PATH).resolve(strict=True)

GPU_ENV = os.environ.copy()
GPU_ENV.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_XET='1',
               TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1')
GPU_PYTHON = Path(GPU_PYTHON).resolve(strict=True)
subprocess.run([str(GPU_PYTHON), str(GPU_SCRIPT), 'environment',
                '--manifest', str(WORK_MANIFEST)], cwd=WORK_SOURCE,
               env=GPU_ENV, check=True, timeout=300)
print('E39 GPU environment checked; assets will be hashed before load')
'''
    notebook["cells"].append(cell(locate, "e39-assets-runtime"))

    execute = '''import numpy as np
import pandas as pd

TRAIN_PATH = COMPETITION_DIR / 'train.csv'
if not TRAIN_PATH.is_file():
    raise FileNotFoundError('Official train.csv must be attached for the text-only proxy')
SMOKE_REPORT = PROBE_ROOT / 'smoke.json'
GPU_SCORES = PROBE_ROOT / 'e39_proxy_logits.npz'
CPU_SCORES = PROBE_ROOT / 'e24_fixed_scores.npz'
PROXY_PARQUET = PROBE_ROOT / 'submission.parquet'
gpu_args = ['--manifest', str(WORK_MANIFEST), '--model-dir', str(BASE_DIR),
            '--adapter-dir', str(ADAPTER_DIR), '--train', str(TRAIN_PATH),
            '--smoke-reference', str(SMOKE_REFERENCE_PATH), '--device-mode', DEVICE_MODE]
subprocess.run([str(GPU_PYTHON), str(GPU_SCRIPT), 'smoke', *gpu_args,
                '--output', str(SMOKE_REPORT)], cwd=WORK_SOURCE, env=GPU_ENV,
               check=True, timeout=3600)

CPU_ENV = os.environ.copy()
CPU_ENV.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false',
               TORCHDYNAMO_DISABLE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS=str(min(4, os.cpu_count() or 4)),
               MKL_NUM_THREADS=str(min(4, os.cpu_count() or 4)))
subprocess.run([
    str(PYTHON_EXECUTABLE), str(CPU_SCRIPT),
    '--full-package', str(PACKAGE_DIR),
    '--modern-seed42', str(MODERN_PACKAGE_DIRS[42]),
    '--modern-seed17', str(MODERN_PACKAGE_DIRS[17]),
    '--modern-seed73', str(MODERN_PACKAGE_DIRS[73]),
    '--e24-package', str(E24_PACKAGE_DIR),
    '--train', str(TRAIN_PATH), '--output', str(CPU_SCORES),
], cwd=WORK_SOURCE, env=CPU_ENV, check=True, timeout=4 * 3600)

subprocess.run([str(GPU_PYTHON), str(GPU_SCRIPT), 'full', *gpu_args,
                '--smoke-report', str(SMOKE_REPORT), '--output', str(GPU_SCORES)],
               cwd=WORK_SOURCE, env=GPU_ENV, check=True, timeout=4 * 3600)

sys.path.insert(0, str(WORK_KAGGLE))
import e39_probe_common as common
with np.load(CPU_SCORES, allow_pickle=False) as e24_data, np.load(GPU_SCORES, allow_pickle=False) as e39_data:
    ids = e24_data['ids']
    if not np.array_equal(ids, e39_data['ids']):
        raise RuntimeError('E24 and E39 proxy ID order differs')
    labels = tuple(e24_data['labels'].tolist())
    if labels != tuple(e39_data['labels'].tolist()) or labels != tuple(asset_manifest['labels']):
        raise RuntimeError('E24 and E39 official label order differs')
    common.write_proxy_submission(PROXY_PARQUET, ids, e24_data['scores'],
                                  e39_data['logits'], labels)
elapsed = time.monotonic() - PROBE_STARTED
if elapsed > TIME_LIMIT_SECONDS:
    raise TimeoutError(f'Full E39 deployment proxy exceeded eight-hour safety budget: {elapsed:.1f}s')
roundtrip = pd.read_parquet(PROXY_PARQUET, engine='pyarrow')
if list(roundtrip.columns) != ['id', 'model'] or len(roundtrip) != 4000:
    raise RuntimeError('Proxy submission Parquet schema/row count changed')
print('E39_UNSCORED_T4_DEPLOYMENT_PROXY_PASS', len(roundtrip), round(elapsed, 1))
print('Do not submit this synthetic-ID proxy Notebook to the competition.')
'''
    notebook["cells"].append(cell(execute, "e39-unscored-full-proxy"))

    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"e39-{index}"
        if item["cell_type"] == "code":
            item["execution_count"] = None
            item["outputs"] = []
    DESTINATION.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
                            .encode("utf-8"))
    print("E39_UNSCORED_NOTEBOOK_READY", DESTINATION)


if __name__ == "__main__":
    main()
