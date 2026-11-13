#!/usr/bin/env python3
"""Split the verified E39 GPU wheelhouse into two stored ZIP uploads.

Each archive remains below 1.9 GB. Together they contain exactly the same
54 flat files as the single-archive package, with no duplicate member names.
This local tool never uploads to Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import package_e39_gpu_wheelhouse as original


ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118_part1.zip"
PART2 = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118_part2.zip"
MAX_ARCHIVE_BYTES = 1_900_000_000


def partition(expected: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    wheels = {name: item for name, item in expected.items() if name.endswith(".whl")}
    if len(wheels) != 52 or len(expected) != 54:
        raise ValueError("E39 split requires exactly 52 wheels and two metadata files")
    first = {name: expected[name] for name in (original.MANIFEST, original.LOCKED)}
    second: dict[str, dict] = {}
    totals = [sum(item["bytes"] for item in first.values()), 0]
    for name, item in sorted(wheels.items(), key=lambda pair: (-pair[1]["bytes"], pair[0])):
        side = 0 if totals[0] <= totals[1] else 1
        (first if side == 0 else second)[name] = item
        totals[side] += item["bytes"]
    if (set(first) & set(second) or set(first) | set(second) != set(expected)
            or original.MANIFEST not in first or original.LOCKED not in first
            or original.MANIFEST in second or original.LOCKED in second):
        raise ValueError("E39 split wheel/member partition is incomplete or repeated")
    if min(totals) <= 0 or max(totals) >= MAX_ARCHIVE_BYTES - 1_000_000:
        raise ValueError(f"E39 split payload exceeds upload safety cap: {totals}")
    return first, second


def build_archive(path: Path, expected: dict[str, dict]) -> dict:
    if path.is_symlink() or path.resolve() != path or path.parent != ROOT / "outputs":
        raise ValueError("E39 split archive path resolves outside ignored outputs")
    if path.exists():
        result = original.verify_archive(path, expected)
        result["status"] = "verified_existing"
        return result
    building = path.with_name(path.name + ".building")
    if building.is_symlink():
        raise ValueError(f"E39 split build path is a symlink: {building.name}")
    if building.exists():
        result = original.verify_archive(building, expected)
        if result["archive_bytes"] >= MAX_ARCHIVE_BYTES:
            raise ValueError(f"Existing E39 split build exceeds upload cap: {building.name}")
        if path.exists():
            raise ValueError(f"Another writer created E39 split archive: {path.name}")
        os.replace(building, path)
        result["status"] = "verified_recovered_build"
        return result
    required = sum(item["bytes"] for item in expected.values())
    if shutil.disk_usage(path.parent).free < required + 2 * 1024**3:
        raise OSError("Insufficient free local space for E39 split ZIP and safety margin")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True, strict_timestamps=True) as archive:
        for name in sorted(expected):
            info = zipfile.ZipInfo(name, date_time=original.FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.file_size = expected[name]["bytes"]
            with (original.SOURCE / name).open("rb") as source, archive.open(
                    info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
    result = original.verify_archive(building, expected)
    if result["archive_bytes"] >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"E39 split ZIP exceeds upload cap: {path.name}")
    if path.exists():
        raise ValueError(f"Another writer created E39 split archive: {path.name}")
    os.replace(building, path)
    result["status"] = "verified_created"
    return result


def verify_combined(first: dict[str, dict], second: dict[str, dict],
                    expected: dict[str, dict]) -> dict:
    result1 = original.verify_archive(PART1, first)
    result2 = original.verify_archive(PART2, second)
    if result1["archive_bytes"] >= MAX_ARCHIVE_BYTES or result2["archive_bytes"] >= MAX_ARCHIVE_BYTES:
        raise ValueError("One E39 split ZIP exceeds 1.9 GB")
    with zipfile.ZipFile(PART1) as zip1, zipfile.ZipFile(PART2) as zip2:
        names1 = set(zip1.namelist())
        names2 = set(zip2.namelist())
    if names1 & names2 or names1 | names2 != set(expected):
        raise ValueError("Combined split ZIP members are duplicated or incomplete")
    if result1["uncompressed_bytes"] + result2["uncompressed_bytes"] != sum(
            item["bytes"] for item in expected.values()):
        raise ValueError("Combined split ZIP payload bytes differ")
    canonical = "".join(f"{name} {expected[name]['bytes']} {expected[name]['sha256']}\n"
                        for name in sorted(expected))
    return {
        "status": "verified_combined",
        "parts": [
            {"path": str(PART1), **result1},
            {"path": str(PART2), **result2},
        ],
        "combined_members": len(names1 | names2),
        "combined_wheels": sum(name.endswith(".whl") for name in names1 | names2),
        "combined_uncompressed_bytes": result1["uncompressed_bytes"] + result2["uncompressed_bytes"],
        "combined_member_manifest_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        "manifest_sha256": expected[original.MANIFEST]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "verify"))
    args = parser.parse_args()
    original.checked_paths()
    expected = original.expected_members()
    first, second = partition(expected)
    if args.action == "package":
        build_archive(PART1, first)
        build_archive(PART2, second)
    print(json.dumps(verify_combined(first, second, expected), sort_keys=True))


if __name__ == "__main__":
    main()
