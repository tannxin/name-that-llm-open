#!/usr/bin/env python3
"""Build an independently hash-checked Kaggle input for three ModernBERT seeds.

The existing E0c/RoBERTa Dataset and CPU wheelhouse are reused unchanged.
This artifact contains only the three full-data ModernBERT packages and their
exact inference source. It never includes competition rows or other outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 17, 73)
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
MODEL_ID = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
RECIPE = "e16_three_seed_modernbert_plus_e0c_roberta_row_z"
MANIFEST_NAME = "e16_artifact_manifest.json"
SOURCE_NAME = "fit_modernbert_full.py"
LABELS = (
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash",
    "kimi-k3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or "\\" in name or path.as_posix() != name or ".." in path.parts:
        raise ValueError(f"Unsafe package artifact path: {name}")
    if len(path.parts) < 2 and name not in ("modernbert_state_dict.pt", "smoke_reference.json"):
        raise ValueError(f"Unexpected package artifact path: {name}")
    return path


def inspect_package(path: Path, seed: int, source_sha: str) -> tuple[dict, list[str]]:
    package = path.resolve(strict=True)
    if not package.is_dir() or any(item.is_symlink() for item in package.rglob("*")):
        raise ValueError(f"Package is missing or contains symlinks: {package}")
    metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "package_schema_version": 1,
        "mode": "full",
        "seed": seed,
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "pretrained_revision": REVISION,
        "train_sha256": TRAIN_SHA256,
        "n_training_rows": 2699,
        "n_training_prompts": 150,
        "labels": list(LABELS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Seed {seed} has unexpected {key}")
    if metadata.get("source_sha256", {}).get(SOURCE_NAME) != source_sha:
        raise ValueError(f"Seed {seed} was fitted with different {SOURCE_NAME} bytes")
    config = metadata.get("training_config", {})
    if any(config.get(key) != value for key, value in (
        ("epochs", 12), ("max_length", 256), ("batch_size", 16),
        ("dropout", 0.1), ("pooling", "mean"),
    )):
        raise ValueError(f"Seed {seed} has unexpected training configuration")
    runtime = metadata.get("runtime_versions", {})
    if runtime.get("torch", "").split("+")[0] != "2.6.0" or runtime.get("transformers") != "4.51.3":
        raise ValueError(f"Seed {seed} was fitted with an unexpected runtime")
    artifact_hashes = metadata.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError(f"Seed {seed} artifact hash manifest is missing")
    expected_names = {"modernbert_state_dict.pt", "smoke_reference.json"}
    actual_files = [item for item in package.rglob("*") if item.is_file()]
    expected_names.update(
        item.relative_to(package).as_posix() for item in actual_files
        if item.relative_to(package).parts[0] in {"encoder_config", "tokenizer"}
    )
    if set(artifact_hashes) != expected_names:
        raise ValueError(f"Seed {seed} artifact manifest does not cover all model files")
    for name, expected_hash in artifact_hashes.items():
        safe_relative(name)
        if sha256(package / name) != expected_hash:
            raise ValueError(f"Seed {seed} model hash mismatch: {name}")
    included = sorted(expected_names | {"metadata.json"})
    smoke_path = package / "smoke_verification.json"
    if not smoke_path.is_file():
        raise ValueError(f"Seed {seed} has no successful smoke verification")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("seed") != seed or smoke.get("package_metadata_sha256") != sha256(package / "metadata.json"):
        raise ValueError(f"Seed {seed} smoke verification does not match metadata")
    if smoke.get("all_predictions_equal") is not True:
        raise ValueError(f"Seed {seed} failed smoke verification")
    difference = smoke.get("max_abs_logit_difference")
    if not isinstance(difference, (int, float)) or not 0 <= difference <= 2e-5:
        raise ValueError(f"Seed {seed} smoke logits exceed the fixed tolerance")
    included.append("smoke_verification.json")
    other_files = {item.relative_to(package).as_posix() for item in actual_files} - set(included)
    if other_files:
        raise ValueError(f"Unexpected files in seed {seed} package: {sorted(other_files)}")
    return metadata, included


def prepare(packages: tuple[Path, Path, Path], output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    paths = tuple(path.resolve(strict=True) for path in packages)
    if len(set(paths)) != len(paths):
        raise ValueError("Three distinct package directories are required")
    source = PROJECT_ROOT / SOURCE_NAME
    source_hash = sha256(source)
    inspected = [inspect_package(path, seed, source_hash) for path, seed in zip(paths, SEEDS, strict=True)]
    if len({metadata["train_sha256"] for metadata, _ in inspected}) != 1:
        raise ValueError("ModernBERT seeds were fitted on different training data")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    shutil.copy2(source, output / "source" / SOURCE_NAME)
    package_names = {}
    for seed, path, (_, included) in zip(SEEDS, paths, inspected, strict=True):
        name = f"modern_seed{seed}"
        package_names[str(seed)] = name
        destination = output / name
        destination.mkdir()
        for relative_name in included:
            copied = destination / relative_name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path / relative_name, copied)
    hashes = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "recipe": RECIPE,
        "source_file": "source/fit_modernbert_full.py",
        "package_dirs": package_names,
        "sha256": hashes,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Prepared {output} ({len(hashes)} files, {total / 1024**2:.1f} MiB)")


def archive(artifact_dir: Path, output: Path) -> None:
    artifact_dir = artifact_dir.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(artifact_dir):
        raise ValueError("ZIP must be new and outside the artifact directory")
    manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = set(manifest["sha256"]) | {MANIFEST_NAME}
    actual = {path.relative_to(artifact_dir).as_posix() for path in artifact_dir.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Artifact directory differs from its manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for name in sorted(expected):
            source = artifact_dir / name
            if name in manifest["sha256"] and sha256(source) != manifest["sha256"][name]:
                raise ValueError(f"Artifact changed after preparation: {name}")
            bundle.write(source, arcname=name)
    with zipfile.ZipFile(output) as bundle:
        if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
            raise ValueError("ZIP entries differ from E16 artifact manifest")
        for name, expected_hash in manifest["sha256"].items():
            with bundle.open(name) as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != expected_hash:
                    raise ValueError(f"ZIP content hash mismatch: {name}")
    print(f"Verified ZIP {output} ({output.stat().st_size / 1024**2:.1f} MiB; SHA-256 {sha256(output)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, help="Optional flat-root ZIP path")
    args = parser.parse_args()
    prepare((args.seed42, args.seed17, args.seed73), args.output)
    if args.zip is not None:
        archive(args.output, args.zip)


if __name__ == "__main__":
    main()
