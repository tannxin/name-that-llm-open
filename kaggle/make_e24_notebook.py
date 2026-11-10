#!/usr/bin/env python3
"""Generate E24 offline Notebook from the successful E16 Notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "submit_name_that_llm_e16.ipynb"
SCRIPT = HERE / "submit_e24_external_e0c.py"
DESTINATION = HERE / "submit_name_that_llm_e24.ipynb"
MANIFEST = "e24_artifact_manifest.json"
RECIPE = "e24_audited_prvsiyan_v9_full_e0c_replacement"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def code_cell(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(value)}


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    script = SCRIPT.read_text(encoding="utf-8")
    script_sha = hashlib.sha256(script.encode("utf-8")).hexdigest()
    notebook["cells"][0]["source"] = lines(
        "# Name That LLM: offline E24 external E0c replacement\n\n"
        "This version changes only the E0c stream from the successful E16 submission. "
        "Its replacement was fitted on all 2,699 official training responses plus "
        "3,620 fixed, provenance-audited Prvsiyan v9 responses covering 17 classes. "
        "The per-class external loss weight totals one quarter of the official "
        "class weight. Qwen has no verified external rows. RoBERTa, three ModernBERT "
        "seeds, row-wise z standardization, and equal stream weights are unchanged. "
        "The fixed 120-prompt fivefold OOF balanced accuracy was 0.425000, versus "
        "0.409259 for E16. This exploratory result is not a Kaggle score: the external "
        "source protocol used global official training length statistics. Attach "
        "the existing E0c/RoBERTa, E16 ModernBERT, and offline wheelhouse inputs, "
        "plus the small E24 artifact. Source credit: "
        "[Prvsiyan Extended Name That LLM Dataset v9](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/version/9), "
        "CC BY 4.0. Internet off; accelerator None.\n"
    )
    notebook["cells"][1]["source"] += lines(
        "\nE24_ARTIFACT_ROOT = None  # Optional mounted E24 Dataset directory override.\n"
        "E24_ARTIFACT_ZIP = None  # Optional E24 ZIP override if autodetection is ambiguous.\n"
    )
    verification = f'''import json
import zipfile

E24_RECIPE = {RECIPE!r}
E24_MANIFEST_NAME = {MANIFEST!r}
if E24_ARTIFACT_ROOT is None:
    candidates = []
    for item in INPUT_ROOT.rglob(E24_MANIFEST_NAME):
        try:
            candidate = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if candidate.get("recipe") == E24_RECIPE:
            candidates.append(item.parent)
    if not candidates:
        zip_candidates = []
        for item in INPUT_ROOT.rglob("*.zip"):
            try:
                with zipfile.ZipFile(item) as bundle:
                    if E24_MANIFEST_NAME in bundle.namelist():
                        zip_candidates.append(item)
            except (OSError, zipfile.BadZipFile):
                continue
        if E24_ARTIFACT_ZIP is not None:
            zip_candidates = [Path(E24_ARTIFACT_ZIP)]
        if len(zip_candidates) != 1:
            raise RuntimeError(f"Expected one E24 artifact ZIP, found {{len(zip_candidates)}}: {{zip_candidates}}")
        extracted = Path("/kaggle/working/name_that_llm_e24_artifact")
        with zipfile.ZipFile(zip_candidates[0]) as bundle:
            names = bundle.namelist()
            if (len(names) != len(set(names)) or E24_MANIFEST_NAME not in names
                    or any(name.startswith("/") or ".." in name.split("/") or "\\\\" in name
                           for name in names)):
                raise RuntimeError("Invalid E24 artifact ZIP")
            bundle.extractall(extracted)
        candidates = [extracted]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one E24 Dataset, found {{len(candidates)}}: {{candidates}}")
    E24_ARTIFACT_ROOT = candidates[0]

E24_ARTIFACT_ROOT = Path(E24_ARTIFACT_ROOT).resolve(strict=True)
e24_manifest = json.loads((E24_ARTIFACT_ROOT / E24_MANIFEST_NAME).read_text(encoding="utf-8"))
if (e24_manifest.get("format_version") != 1 or e24_manifest.get("recipe") != E24_RECIPE
        or e24_manifest.get("package_dir") != "e24_e0c"
        or e24_manifest.get("source_dir") != "source"):
    raise RuntimeError("Unexpected E24 artifact manifest")
e24_hashes = e24_manifest.get("sha256")
if not isinstance(e24_hashes, dict) or set(e24_hashes) != {{
    "e24_e0c/metadata.json", "e24_e0c/e0c.joblib", "e24_e0c/smoke_reference.json",
    "source/fit_e24_full_e0c.py", "source/run_e24_prvsiyan_external_ablation.py",
}}:
    raise RuntimeError("Unexpected E24 artifact file list")
actual_files = {{p.relative_to(E24_ARTIFACT_ROOT).as_posix()
                for p in E24_ARTIFACT_ROOT.rglob("*") if p.is_file()}}
if actual_files != set(e24_hashes) | {{E24_MANIFEST_NAME}}:
    raise RuntimeError("E24 artifact contents differ from manifest")
for name, digest in e24_hashes.items():
    path = (E24_ARTIFACT_ROOT / name).resolve(strict=True)
    if not path.is_relative_to(E24_ARTIFACT_ROOT) or sha256_file(path) != digest:
        raise RuntimeError(f"E24 artifact hash mismatch: {{name}}")
E24_PACKAGE_DIR = (E24_ARTIFACT_ROOT / "e24_e0c").resolve(strict=True)
E24_SOURCE_DIR = (E24_ARTIFACT_ROOT / "source").resolve(strict=True)
trained = json.loads((E24_PACKAGE_DIR / "metadata.json").read_text(encoding="utf-8"))
for name in ("fit_e24_full_e0c.py", "run_e24_prvsiyan_external_ablation.py"):
    if sha256_file(E24_SOURCE_DIR / name) != trained.get("source_sha256", {{}}).get(name):
        raise RuntimeError(f"E24 fitted source mismatch: {{name}}")
for name in ("run_e0.py", "run_style.py", "run_combined.py"):
    if sha256_file(SOURCE_DIR / name) != trained.get("source_sha256", {{}}).get(name):
        raise RuntimeError(f"E24 shared E0c source mismatch: {{name}}")
print("Verified small E24 E0c package/source hashes:", E24_ARTIFACT_ROOT)
'''
    notebook["cells"].insert(4, code_cell(verification))
    execution = f'''import hashlib
import os
import shutil
import subprocess

WORK_SOURCE = Path("/kaggle/working/name_that_llm_source_e24")
shutil.copytree(SOURCE_DIR, WORK_SOURCE, dirs_exist_ok=True)
shutil.copy2(MODERN_SOURCE_PATH, WORK_SOURCE / "fit_modernbert_full.py")
for name in ("fit_e24_full_e0c.py", "run_e24_prvsiyan_external_ablation.py"):
    shutil.copy2(E24_SOURCE_DIR / name, WORK_SOURCE / name)
WORK_SCRIPT = WORK_SOURCE / "kaggle" / "submit_e24_external_e0c.py"
WORK_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
WORK_SCRIPT.write_text({script!r}, encoding="utf-8")
if hashlib.sha256(WORK_SCRIPT.read_bytes()).hexdigest() != "{script_sha}":
    raise RuntimeError("E24 submission script hash mismatch")

if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()
environment = os.environ.copy()
environment["HF_HUB_OFFLINE"] = "1"
environment["HF_HUB_DISABLE_XET"] = "1"
environment["TRANSFORMERS_OFFLINE"] = "1"
environment["TOKENIZERS_PARALLELISM"] = "false"
environment["TORCHDYNAMO_DISABLE"] = "1"
environment["CUDA_VISIBLE_DEVICES"] = ""
threads = str(min(4, os.cpu_count() or 4))
environment["OMP_NUM_THREADS"] = threads
environment["MKL_NUM_THREADS"] = threads
command = [
    str(PYTHON_EXECUTABLE), str(WORK_SCRIPT),
    "--full-package", str(PACKAGE_DIR),
    "--modern-seed42", str(MODERN_PACKAGE_DIRS[42]),
    "--modern-seed17", str(MODERN_PACKAGE_DIRS[17]),
    "--modern-seed73", str(MODERN_PACKAGE_DIRS[73]),
    "--e24-package", str(E24_PACKAGE_DIR),
    "--test", str(TEST_PATH),
    "--output", str(OUTPUT_PATH),
]
subprocess.run(command, cwd=WORK_SOURCE, env=environment, check=True, timeout=8 * 60 * 60)
if not OUTPUT_PATH.is_file():
    raise RuntimeError("Inference completed without submission.parquet")
subprocess.run(
    [str(PYTHON_EXECUTABLE), "-c",
     "import sys, pyarrow.parquet as pq; print('Ready:', sys.argv[1], 'rows:', pq.read_metadata(sys.argv[1]).num_rows)",
     str(OUTPUT_PATH)], check=True,
)
'''
    notebook["cells"][-1]["source"] = lines(execution)
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"e24-{index}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {DESTINATION} (submission source SHA-256 {script_sha})")


if __name__ == "__main__":
    main()
