#!/usr/bin/env python3
"""Create and verify a flat-root ZIP of a prepared Kaggle artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def archive(artifact_dir: Path, output: Path) -> None:
    artifact_dir = artifact_dir.resolve(strict=True)
    output = output.resolve()
    if not artifact_dir.is_dir():
        raise NotADirectoryError(artifact_dir)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if output.is_relative_to(artifact_dir):
        raise ValueError("ZIP output must be outside the artifact directory")
    manifest = json.loads(
        (artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    expected = set(manifest["sha256"]) | {"artifact_manifest.json"}
    files = [path for path in artifact_dir.rglob("*") if path.is_file()]
    names = {path.relative_to(artifact_dir).as_posix() for path in files}
    if names != expected or any(path.is_symlink() for path in files):
        raise ValueError("Artifact directory differs from its manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(files):
            bundle.write(path, arcname=path.relative_to(artifact_dir).as_posix())
    with zipfile.ZipFile(output) as bundle:
        if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
            raise ValueError("ZIP entries differ from the artifact manifest")
        for name, expected_hash in manifest["sha256"].items():
            with bundle.open(name) as stream:
                if sha256_stream(stream) != expected_hash:
                    raise ValueError(f"ZIP content hash mismatch: {name}")
    with output.open("rb") as stream:
        output_hash = sha256_stream(stream)
    print(
        f"Verified {output} ({output.stat().st_size / 1024**2:.1f} MiB, "
        f"{len(expected)} files, SHA-256 {output_hash})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
