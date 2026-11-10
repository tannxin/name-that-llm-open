#!/usr/bin/env python3
"""Build a hash-checked Kaggle input directory from the fitted full package.

This copies only the five inference source files and the fitted model package.
Competition data and experiment outputs are never included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "final_pipeline.py",
    "run_e0.py",
    "run_style.py",
    "run_combined.py",
    "run_fixed_blend.py",
)
RECIPE = "e0c_plus_roberta_12ep_scalerfix_equal_rank"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(full_package: Path, output: Path) -> None:
    full_package = full_package.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    metadata_path = full_package / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mode") != "full" or metadata.get("recipe") != RECIPE:
        raise ValueError("Expected the fitted full package for the selected recipe")
    expected_sources = metadata.get("source_sha256")
    if not isinstance(expected_sources, dict) or set(expected_sources) != set(SOURCE_FILES):
        raise ValueError("Package has an unexpected source hash manifest")
    for name in SOURCE_FILES:
        if sha256(PROJECT_ROOT / name) != expected_sources[name]:
            raise ValueError(f"Source differs from the fitted package: {name}")

    for name, key in (
        ("e0c.joblib", "e0c_joblib_sha256"),
        ("roberta_state_dict.pt", "roberta_state_sha256"),
    ):
        if sha256(full_package / name) != metadata.get(key):
            raise ValueError(f"Model artifact differs from fitted metadata: {name}")
    for directory in ("tokenizer", "encoder_config"):
        path = full_package / directory
        if not path.is_dir() or not any(path.iterdir()):
            raise ValueError(f"Package is missing {directory}")
    if any(path.is_symlink() for path in full_package.rglob("*")):
        raise ValueError("Package contains symlinks; resolve the package before upload")

    output.mkdir(parents=True)
    source_dir = output / "source"
    source_dir.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(PROJECT_ROOT / name, source_dir / name)
    shutil.copy2(PROJECT_ROOT / "requirements-final.txt", output / "requirements-final.txt")
    shutil.copytree(full_package, output / "full_package")

    hashes = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "recipe": RECIPE,
        "source_dir": "source",
        "package_dir": "full_package",
        "sha256": hashes,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    total_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Prepared {output} ({len(hashes)} files, {total_size / 1024**2:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.full_package, args.output)


if __name__ == "__main__":
    main()
