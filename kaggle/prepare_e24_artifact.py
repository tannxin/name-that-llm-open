#!/usr/bin/env python3
"""Build the small hash-checked E24 E0c replacement input for Kaggle.

The existing E16 input continues to supply RoBERTa, ModernBERT, and the
offline wheelhouse. This ZIP contains no official or external text rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = "e24_audited_prvsiyan_v9_full_e0c_replacement"
MANIFEST = "e24_artifact_manifest.json"
SOURCE_FILES = ("fit_e24_full_e0c.py", "run_e24_prvsiyan_external_ablation.py")
SHARED_FILES = ("run_e0.py", "run_style.py", "run_combined.py")
MODEL_FILES = ("metadata.json", "e0c.joblib", "smoke_reference.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(package: Path, output: Path) -> None:
    package = package.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if {path.name for path in package.iterdir()} != set(MODEL_FILES):
        raise ValueError("E24 full E0c package has unexpected files")
    metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "package_schema_version": 1,
        "recipe": RECIPE,
        "train_sha256": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
        "n_official_rows": 2699,
        "n_official_prompts": 150,
        "n_external_rows": 3620,
        "n_external_prompts": 250,
        "external_to_official_weight_ratio": 0.25,
        "selected_manifest_sha256": "d7f8b7c9b0480ec896777e667589bf7d2fe12e503aa8f867c0708d0656ad6191",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"E24 package has unexpected {key}")
    if metadata.get("sklearn_version") != "1.7.2":
        raise ValueError("E24 package must use original scikit-learn 1.7.2")
    source_hashes = metadata.get("source_sha256", {})
    if set(source_hashes) != set(SOURCE_FILES + SHARED_FILES):
        raise ValueError("E24 source file list changed")
    for name in SOURCE_FILES + SHARED_FILES:
        if sha256(ROOT / name) != source_hashes[name]:
            raise ValueError(f"Fitted source hash mismatch: {name}")
    for name in MODEL_FILES[1:]:
        if sha256(package / name) != metadata.get("artifact_sha256", {}).get(name):
            raise ValueError(f"Fitted artifact hash mismatch: {name}")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    (output / "e24_e0c").mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, output / "source" / name)
    for name in MODEL_FILES:
        shutil.copy2(package / name, output / "e24_e0c" / name)
    hashes = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "recipe": RECIPE,
        "package_dir": "e24_e0c",
        "source_dir": "source",
        "sha256": hashes,
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def archive(output: Path, zip_path: Path) -> None:
    output = output.resolve(strict=True)
    zip_path = zip_path.resolve()
    if zip_path.exists() or zip_path.is_relative_to(output):
        raise ValueError("ZIP path must be new and outside artifact directory")
    manifest = json.loads((output / MANIFEST).read_text(encoding="utf-8"))
    expected = set(manifest["sha256"]) | {MANIFEST}
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != expected:
        raise ValueError("E24 artifact directory differs from manifest")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for name in sorted(expected):
            source = output / name
            if name in manifest["sha256"] and sha256(source) != manifest["sha256"][name]:
                raise ValueError(f"E24 artifact changed: {name}")
            bundle.write(source, arcname=name)
    with zipfile.ZipFile(zip_path) as bundle:
        if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
            raise ValueError("E24 ZIP entries differ from manifest")
        for name, expected_hash in manifest["sha256"].items():
            with bundle.open(name) as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ValueError(f"E24 ZIP entry hash mismatch: {name}")
    print(f"Verified {zip_path} ({zip_path.stat().st_size} bytes; SHA-256 {sha256(zip_path)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("outputs/e24_full_e0c_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/e24_kaggle_artifact_v1"))
    parser.add_argument("--zip", type=Path, default=Path("outputs/e24_kaggle_artifact_v1.zip"))
    args = parser.parse_args()
    prepare(args.package, args.output)
    archive(args.output, args.zip)


if __name__ == "__main__":
    main()
