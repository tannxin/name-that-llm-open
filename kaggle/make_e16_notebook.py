#!/usr/bin/env python3
"""Generate a separate offline E16 Kaggle Notebook from the proven v3 setup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "submit_name_that_llm_zscore.ipynb"
DESTINATION = HERE / "submit_name_that_llm_e16.ipynb"
SCRIPT = HERE / "submit_e16_three_seed.py"
MANIFEST = "e16_artifact_manifest.json"
RECIPE = "e16_three_seed_modernbert_plus_e0c_roberta_row_z"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def code_cell(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    script = SCRIPT.read_text(encoding="utf-8")
    script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
    notebook["cells"][0]["source"] = lines(
        "# Name That LLM: offline E16 three-family submission\n\n"
        "This candidate keeps the verified full-data E0c and 12-epoch RoBERTa packages "
        "from the public 0.401 submission, and adds three full-data ModernBERT-base "
        "seeds (42, 17, 73). Each model's 18 logits are standardized across classes "
        "within a row. The three ModernBERT standardized scores are averaged into "
        "one stream; that stream is added equally to E0c and RoBERTa. The rule was "
        "fixed on development OOF (balanced accuracy 0.409259 versus 0.399537 for "
        "E0c+RoBERTa). The prior locked holdout was not used to select it. Attach "
        "the existing private E0c/RoBERTa Dataset, the separate E16 ModernBERT "
        "Dataset, and the existing offline CPU wheelhouse. Set accelerator None and "
        "Internet off. The hidden rerun writes `submission.parquet` from the official "
        "test CSV.\n"
    )
    notebook["cells"][1]["source"] += lines(
        "\nMODERN_ARTIFACT_ROOT = None  # Optional mounted E16 Dataset directory override.\n"
        "MODERN_ARTIFACT_ZIP = None  # Optional E16 ZIP override if autodetection is ambiguous.\n"
    )
    modern_verification = f'''import json
import zipfile

MODERN_RECIPE = {RECIPE!r}
MODERN_MANIFEST_NAME = {MANIFEST!r}
MODERN_ARTIFACT_ARCHIVE = None

if MODERN_ARTIFACT_ROOT is None:
    candidates = []
    for item in INPUT_ROOT.rglob(MODERN_MANIFEST_NAME):
        try:
            candidate = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if candidate.get("recipe") == MODERN_RECIPE:
            candidates.append(item.parent)
    if not candidates:
        zip_candidates = []
        for item in INPUT_ROOT.rglob("*.zip"):
            try:
                with zipfile.ZipFile(item) as bundle:
                    if MODERN_MANIFEST_NAME in bundle.namelist():
                        zip_candidates.append(item)
            except (OSError, zipfile.BadZipFile):
                continue
        if MODERN_ARTIFACT_ZIP is not None:
            zip_candidates = [Path(MODERN_ARTIFACT_ZIP)]
        if len(zip_candidates) != 1:
            raise RuntimeError(
                f"Expected one E16 artifact ZIP, found {{len(zip_candidates)}}: {{zip_candidates}}. "
                "Set MODERN_ARTIFACT_ZIP in the config cell."
            )
        MODERN_ARTIFACT_ARCHIVE = Path(zip_candidates[0]).resolve(strict=True)
        extracted_root = Path("/kaggle/working/name_that_llm_e16_artifact")
        with zipfile.ZipFile(MODERN_ARTIFACT_ARCHIVE) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or MODERN_MANIFEST_NAME not in names:
                raise RuntimeError("Invalid E16 artifact ZIP")
            if any(name.startswith("/") or ".." in name.split("/") or "\\\\" in name for name in names):
                raise RuntimeError("Unsafe path in E16 artifact ZIP")
            bundle.extractall(extracted_root)
        candidates = [extracted_root]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one E16 Dataset, found {{len(candidates)}}: {{candidates}}. "
            "Set MODERN_ARTIFACT_ROOT in the config cell."
        )
    MODERN_ARTIFACT_ROOT = candidates[0]

MODERN_ARTIFACT_ROOT = Path(MODERN_ARTIFACT_ROOT).resolve(strict=True)
modern_manifest = json.loads(
    (MODERN_ARTIFACT_ROOT / MODERN_MANIFEST_NAME).read_text(encoding="utf-8")
)
if modern_manifest.get("format_version") != 1 or modern_manifest.get("recipe") != MODERN_RECIPE:
    raise RuntimeError("Unexpected E16 artifact manifest")
if modern_manifest.get("source_file") != "source/fit_modernbert_full.py":
    raise RuntimeError("Unexpected E16 source path")
expected_package_dirs = {{"42": "modern_seed42", "17": "modern_seed17", "73": "modern_seed73"}}
if modern_manifest.get("package_dirs") != expected_package_dirs:
    raise RuntimeError("Unexpected E16 package directory mapping")
modern_hashes = modern_manifest.get("sha256")
if not isinstance(modern_hashes, dict) or not modern_hashes:
    raise RuntimeError("E16 artifact file hash manifest is missing")
actual_files = {{item.relative_to(MODERN_ARTIFACT_ROOT).as_posix()
                for item in MODERN_ARTIFACT_ROOT.rglob("*") if item.is_file()}}
if actual_files != set(modern_hashes) | {{MODERN_MANIFEST_NAME}}:
    raise RuntimeError("E16 artifact files differ from manifest")
for relative_name, expected_hash in modern_hashes.items():
    path = (MODERN_ARTIFACT_ROOT / relative_name).resolve(strict=True)
    if not path.is_relative_to(MODERN_ARTIFACT_ROOT) or sha256_file(path) != expected_hash:
        raise RuntimeError(f"E16 artifact hash mismatch: {{relative_name}}")

MODERN_SOURCE_PATH = (MODERN_ARTIFACT_ROOT / modern_manifest["source_file"]).resolve(strict=True)
MODERN_PACKAGE_DIRS = {{
    int(seed): (MODERN_ARTIFACT_ROOT / directory).resolve(strict=True)
    for seed, directory in expected_package_dirs.items()
}}
for seed, package in MODERN_PACKAGE_DIRS.items():
    if not package.is_relative_to(MODERN_ARTIFACT_ROOT):
        raise RuntimeError(f"E16 seed {{seed}} package escapes Dataset root")
    trained = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    if trained.get("seed") != seed:
        raise RuntimeError(f"E16 seed directory/metadata mismatch: {{seed}}")
    if trained.get("source_sha256", {{}}).get("fit_modernbert_full.py") != sha256_file(MODERN_SOURCE_PATH):
        raise RuntimeError(f"E16 seed {{seed}} source differs from fitted source")
    smoke = json.loads((package / "smoke_verification.json").read_text(encoding="utf-8"))
    if (smoke.get("seed") != seed or smoke.get("all_predictions_equal") is not True
            or smoke.get("package_metadata_sha256") != sha256_file(package / "metadata.json")
            or not isinstance(smoke.get("max_abs_logit_difference"), (int, float))
            or not 0 <= smoke["max_abs_logit_difference"] <= 2e-5):
        raise RuntimeError(f"E16 seed {{seed}} has no valid reload smoke verification")
print("Verified E16 ModernBERT package/source hashes:", MODERN_ARTIFACT_ROOT)
'''
    notebook["cells"].insert(3, code_cell(modern_verification))

    offline_probe = '''# Probe the ModernBERT config and tokenizer with the interpreter selected above.
probe_environment = os.environ.copy()
probe_environment["HF_HUB_OFFLINE"] = "1"
probe_environment["TRANSFORMERS_OFFLINE"] = "1"
probe_environment["HF_HUB_DISABLE_XET"] = "1"
probe_environment["TORCHDYNAMO_DISABLE"] = "1"
probe = (
    "import sys, transformers; from transformers import AutoConfig, AutoTokenizer; "
    "c=AutoConfig.from_pretrained(sys.argv[1], local_files_only=True); "
    "assert c.model_type == 'modernbert' and c.hidden_size == 768; "
    "AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True, use_fast=True); "
    "print('Local ModernBERT config and tokenizer ready:', transformers.__version__)"
)
subprocess.run(
    [str(PYTHON_EXECUTABLE), "-c", probe,
     str(MODERN_PACKAGE_DIRS[42] / "encoder_config"),
     str(MODERN_PACKAGE_DIRS[42] / "tokenizer")],
    env=probe_environment, check=True,
)
'''
    notebook["cells"].insert(5, code_cell(offline_probe))

    execution = f'''import hashlib
import os
import shutil
import subprocess


WORK_SOURCE = Path("/kaggle/working/name_that_llm_source_e16")
shutil.copytree(SOURCE_DIR, WORK_SOURCE, dirs_exist_ok=True)
shutil.copy2(MODERN_SOURCE_PATH, WORK_SOURCE / "fit_modernbert_full.py")
WORK_SCRIPT = WORK_SOURCE / "kaggle" / "submit_e16_three_seed.py"
WORK_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
WORK_SCRIPT.write_text({script!r}, encoding="utf-8")
if hashlib.sha256(WORK_SCRIPT.read_bytes()).hexdigest() != "{script_hash}":
    raise RuntimeError("E16 submission script hash mismatch")

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
    "--test", str(TEST_PATH),
    "--output", str(OUTPUT_PATH),
]
subprocess.run(command, cwd=WORK_SOURCE, env=environment, check=True, timeout=8 * 60 * 60)
if not OUTPUT_PATH.is_file():
    raise RuntimeError("Inference completed without submission.parquet")
subprocess.run(
    [str(PYTHON_EXECUTABLE), "-c",
     "import sys, pyarrow.parquet as pq; print('Ready:', sys.argv[1], 'rows:', pq.read_metadata(sys.argv[1]).num_rows)",
     str(OUTPUT_PATH)],
    check=True,
)
'''
    notebook["cells"][-1]["source"] = lines(execution)
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"e16-{index}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {DESTINATION} with E16 submission script SHA-256 {script_hash}")


if __name__ == "__main__":
    main()
