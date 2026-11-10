#!/usr/bin/env python3
"""Build the offline z-score Notebook from the verified rank Notebook and script."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "submit_name_that_llm.ipynb"
DESTINATION = HERE / "submit_name_that_llm_zscore.ipynb"
SCRIPT = HERE / "submit_normalized.py"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    script = SCRIPT.read_text(encoding="utf-8")
    script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
    notebook["cells"][0]["source"] = lines(
        "# Name That LLM: offline z-score fusion submission\n\n"
        "This Notebook reuses the verified full-data E0c and 12-epoch RoBERTa package from the original submission. "
        "Only the fixed score fusion changes: each model's 18 class scores are standardized within a row, "
        "then added with equal weight. On the original 120-prompt five-fold development set, this gave "
        "0.3995 balanced accuracy versus 0.3833 for equal ranks. The reserved holdout was not used to select this rule. "
        "The existing private model Dataset and CPU wheelhouse remain attached. Internet is off. "
        "The code writes `submission.parquet` from the official hidden `test.csv` when Kaggle reruns it.\n"
    )
    execution = f'''import hashlib
import os
import shutil
import subprocess


# The old model artifact is immutable; copy only its five verified source files
# into a writable directory and add the exact source of the new fusion script.
WORK_SOURCE = Path("/kaggle/working/name_that_llm_source_zscore")
shutil.copytree(SOURCE_DIR, WORK_SOURCE, dirs_exist_ok=True)
WORK_SCRIPT = WORK_SOURCE / "kaggle" / "submit_normalized.py"
WORK_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
WORK_SCRIPT.write_text({script!r}, encoding="utf-8")
if hashlib.sha256(WORK_SCRIPT.read_bytes()).hexdigest() != "{script_hash}":
    raise RuntimeError("Normalized submission script hash mismatch")

if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()
environment = os.environ.copy()
environment["HF_HUB_OFFLINE"] = "1"
environment["HF_HUB_DISABLE_XET"] = "1"
environment["TRANSFORMERS_OFFLINE"] = "1"
environment["TOKENIZERS_PARALLELISM"] = "false"
threads = str(min(4, os.cpu_count() or 4))
environment["OMP_NUM_THREADS"] = threads
environment["MKL_NUM_THREADS"] = threads
command = [
    str(PYTHON_EXECUTABLE), str(WORK_SCRIPT),
    "--full-package", str(PACKAGE_DIR),
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
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {DESTINATION} with normalized script SHA-256 {script_hash}")


if __name__ == "__main__":
    main()
