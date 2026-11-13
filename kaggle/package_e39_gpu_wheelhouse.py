#!/usr/bin/env python3
"""Package the verified E39 CUDA wheelhouse as a minimal, stored ZIP.

The archive contains only the 52 wheel files named by the final manifest,
the manifest itself, and its SHA-locked requirements file. It stays below
Git-ignored outputs/ and is never uploaded by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import prepare_e39_gpu_wheelhouse as wheelhouse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118"
ARCHIVE = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118.zip"
MANIFEST = "e39_gpu_wheel_manifest.json"
LOCKED = "requirements-e39-gpu-locked.txt"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
EXPECTED_MANIFEST_SHA256 = "a9a373ba8cf9d688b7730bdef811d4bd578416bda66bc1200f35af6accc881b8"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_paths() -> None:
    ignored_root = ROOT / "outputs"
    if ignored_root.is_symlink() or SOURCE.is_symlink() or ARCHIVE.is_symlink():
        raise ValueError("E39 packaging path must not be a symlink")
    if SOURCE.resolve() != SOURCE or ARCHIVE.resolve() != ARCHIVE:
        raise ValueError("E39 packaging path resolves outside its expected location")
    if not SOURCE.is_dir() or not ignored_root.is_dir():
        raise FileNotFoundError("Git-ignored E39 wheelhouse does not exist")


def expected_members() -> dict[str, dict]:
    if sha256_path(SOURCE / MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("E39 manifest bytes differ from the reviewed Notebook lock")
    manifest = wheelhouse.verify(wheelhouse.load_spec(), SOURCE)
    rows = manifest["wheels"]
    if len(rows) != 52 or len(manifest["selected_packages"]) != 47:
        raise ValueError("Final E39 GPU wheelhouse count differs from reviewed lock")
    expected = {}
    for row in rows:
        name = row["file"]
        if (Path(name).name != name or not name.endswith(".whl") or
                name in expected or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError(f"Unsafe or repeated E39 wheel member: {name}")
        expected[name] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    expected[MANIFEST] = {"bytes": (SOURCE / MANIFEST).stat().st_size,
                          "sha256": sha256_path(SOURCE / MANIFEST)}
    expected[LOCKED] = {"bytes": (SOURCE / LOCKED).stat().st_size,
                        "sha256": manifest["locked_requirements_sha256"]}
    for name in expected:
        path = SOURCE / name
        if path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError(f"E39 source member is missing or redirects: {name}")
    return expected


def verify_archive(path: Path, expected: dict[str, dict]) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("E39 archive is absent or a symlink")
    with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if (len(names) != len(expected) or set(names) != set(expected) or
                names != sorted(expected)):
            raise ValueError("E39 ZIP member list differs from final manifest")
        for item in members:
            name = item.filename
            entry = expected[name]
            if (item.is_dir() or item.compress_type != zipfile.ZIP_STORED or
                    item.flag_bits & 1 or item.file_size != entry["bytes"] or
                    item.date_time != FIXED_TIME or item.create_system != 3 or
                    item.external_attr != 0o644 << 16 or item.comment):
                raise ValueError(f"E39 ZIP metadata/size differs: {name}")
            digest = hashlib.sha256()
            count = 0
            with archive.open(item, "r") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    count += len(block)
                    digest.update(block)
            if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"E39 ZIP member SHA/size differs: {name}")
    return {"members": len(expected), "wheel_members": len(expected) - 2,
            "uncompressed_bytes": sum(entry["bytes"] for entry in expected.values()),
            "archive_bytes": path.stat().st_size, "archive_sha256": sha256_path(path),
            "manifest_sha256": expected[MANIFEST]["sha256"]}


def package_archive(expected: dict[str, dict]) -> dict:
    if ARCHIVE.exists():
        result = verify_archive(ARCHIVE, expected)
        result["status"] = "verified_existing"
        return result
    building = ARCHIVE.with_name(ARCHIVE.name + ".building")
    if building.is_symlink() or building.exists():
        raise ValueError("An unfinished E39 archive exists; inspect it before retrying")
    need = sum(entry["bytes"] for entry in expected.values())
    if shutil.disk_usage(ROOT / "outputs").free < need + 2 * 1024**3:
        raise OSError("Insufficient local free space for stored E39 ZIP and safety margin")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True, strict_timestamps=True) as archive:
        for name in sorted(expected):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.file_size = expected[name]["bytes"]
            with (SOURCE / name).open("rb") as source, archive.open(
                    info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
    result = verify_archive(building, expected)
    if ARCHIVE.exists():
        raise ValueError("Another writer created the E39 archive")
    os.replace(building, ARCHIVE)
    result["status"] = "verified_created"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "verify"))
    args = parser.parse_args()
    checked_paths()
    expected = expected_members()
    if args.action == "package":
        result = package_archive(expected)
    else:
        result = verify_archive(ARCHIVE, expected)
        result["status"] = "verified_existing"
    result["path"] = str(ARCHIVE)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
