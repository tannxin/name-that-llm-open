#!/usr/bin/env python3
"""Fetch only the pinned E39 Linux CUDA dependency wheels.

The checked-in pins were read from official PyPI release JSON. Each run also
checks the exact file, size and SHA-256 against that JSON. Aliyun is a byte
transfer source, never a trust source. This does not contact Kaggle or use any
competition data, and writes only below Git-ignored outputs/.

Run only after the existing pip wheelhouse writer has exited: pip does not
honor this script's lock. A stale lock after a crash requires manual process
inspection before removal.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "kaggle/e39_pypi_cuda_wheel_pins_v1.json"
SPEC_PATH = ROOT / "kaggle/e39_gpu_wheelhouse_spec.json"
DEST = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118"
PART_BYTES = 8 * 1024 * 1024
ALLOWED = frozenset({
    "nvidia-cuda-nvrtc-cu11", "nvidia-cuda-runtime-cu11",
    "nvidia-cuda-cupti-cu11", "nvidia-cudnn-cu11",
    "nvidia-cublas-cu11", "nvidia-cufft-cu11", "nvidia-curand-cu11",
    "nvidia-cusolver-cu11", "nvidia-cusparse-cu11",
    "nvidia-nccl-cu11", "nvidia-nvtx-cu11", "triton",
    "bitsandbytes", "peft", "accelerate",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file_path(path: Path) -> None:
    """Reject an existing link/junction before using an expected output name."""
    if path.is_symlink() or path.resolve() != path:
        raise ValueError(f"Output path resolves away from its expected name: {path.name}")


def load_pins() -> list[dict]:
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    if (pins.get("schema_version") != 1
            or pins.get("trust_source") != "https://pypi.org/pypi/{distribution}/{version}/json"
            or pins.get("transfer_source") != "https://mirrors.aliyun.com/pypi"):
        raise ValueError("E39 auxiliary wheel pin schema or sources changed")
    rows = pins.get("wheels")
    if not isinstance(rows, list) or len(rows) != len(ALLOWED):
        raise ValueError("E39 auxiliary wheel set changed")
    found = set()
    for row in rows:
        name, version, filename = row["distribution"], row["version"], row["file"]
        if name not in ALLOWED or name in found:
            raise ValueError("Unexpected or repeated E39 auxiliary distribution")
        found.add(name)
        if (not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version)
                or not filename.startswith(name.replace("-", "_") + "-" + version + "-")
                or Path(filename).name != filename or not filename.endswith(".whl")
                or not re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", filename)
                or not isinstance(row["bytes"], int) or row["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError(f"Malformed E39 wheel pin: {name}")
        if name in ALLOWED - {"peft", "accelerate"} and "x86_64.whl" not in filename:
            raise ValueError(f"Expected an x86_64 Linux wheel: {name}")
    if found != ALLOWED:
        raise ValueError("E39 auxiliary distribution set changed")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = set(spec["linux_torch_dependencies"])
    expected.update(line for line in spec["requirements"]
                    if line.split("==", 1)[0] in {"bitsandbytes", "peft", "accelerate"})
    actual = {row["distribution"] + "==" + row["version"] for row in rows}
    if actual != expected or len(expected) != len(ALLOWED):
        raise ValueError("Pinned CUDA/Triton/auxiliary wheels differ from E39 runtime spec")
    return rows


def official_mirror_url(pin: dict) -> str:
    """Recheck pinned identity against official PyPI JSON; derive mirror path."""
    name, version = pin["distribution"], pin["version"]
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    request = urllib.request.Request(metadata_url, headers={
        "Accept": "application/json", "User-Agent": "ntl-e39-wheel-fetch/1",
    })
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise ValueError(f"Official PyPI status {response.status}: {name}")
                final_url = urllib.parse.urlsplit(response.geturl())
                if final_url.scheme != "https" or final_url.hostname != "pypi.org":
                    raise ValueError(f"Official PyPI metadata redirected away from pypi.org: {name}")
                payload = response.read(5 * 1024 * 1024 + 1)
            if len(payload) > 5 * 1024 * 1024:
                raise ValueError(f"Official PyPI JSON is unexpectedly large: {name}")
            metadata = json.loads(payload)
            break
        except (OSError, http.client.HTTPException) as exc:
            if attempt == 3:
                raise RuntimeError(f"Official PyPI metadata unavailable: {name}") from exc
            time.sleep(2 * attempt)
    if (metadata["info"]["version"] != version or
            canonicalize_name(metadata["info"]["name"]) != name):
        raise ValueError(f"Official PyPI package identity differs: {name}")
    matches = [row for row in metadata["urls"] if row["filename"] == pin["file"]]
    if len(matches) != 1:
        raise ValueError(f"Exact wheel absent from official PyPI JSON: {name}")
    row = matches[0]
    if (row["packagetype"] != "bdist_wheel" or row["size"] != pin["bytes"]
            or row["digests"]["sha256"] != pin["sha256"] or row.get("yanked")):
        raise ValueError(f"Official PyPI identity, size or SHA differs: {name}")
    origin = urllib.parse.urlsplit(row["url"])
    if (origin.scheme != "https" or origin.hostname != "files.pythonhosted.org"
            or origin.username is not None or origin.password is not None
            or not origin.path.startswith("/packages/")
            or urllib.parse.unquote(origin.path.rsplit("/", 1)[-1]) != pin["file"]
            or origin.query or origin.fragment):
        raise ValueError(f"Unexpected official PyPI file URL: {name}")
    return urllib.parse.urlunsplit((
        "https", "mirrors.aliyun.com", "/pypi" + origin.path, "", ""))


def fetch_part(url: str, part_dir: Path, index: int, start: int,
               end: int, total: int) -> tuple[int, int]:
    target = part_dir / f"part_{index:04d}"
    check_file_path(target)
    expected = end - start + 1
    if target.is_file():
        if target.stat().st_size == expected:
            return index, expected
        raise ValueError(f"Existing segment has wrong length: {target.name}")
    temporary = part_dir / f"part_{index:04d}.partial"
    for attempt in range(1, 4):
        check_file_path(temporary)
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={
                "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                "User-Agent": "ntl-e39-wheel-fetch/1",
            })
            with urllib.request.urlopen(request, timeout=120) as response:
                if (response.status != 206 or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{total}"):
                    raise ValueError(f"Mirror ignored or changed range {index}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected:
                    raise ValueError(f"Mirror Content-Length differs: segment {index}")
                count = 0
                with temporary.open("xb") as output:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        count += len(block)
                        if count > expected:
                            raise ValueError(f"Mirror overran segment {index}")
                        output.write(block)
            if count != expected:
                raise ValueError(f"Mirror truncated segment {index}")
            os.replace(temporary, target)
            return index, count
        except (OSError, ValueError, http.client.HTTPException):
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("Unreachable")


def fetch_wheel(pin: dict, dest: Path, segments_root: Path,
                workers: int) -> None:
    filename = pin["file"]
    final = dest / filename
    check_file_path(final)
    url = official_mirror_url(pin)
    if final.is_file():
        if final.stat().st_size != pin["bytes"] or sha256(final) != pin["sha256"]:
            raise ValueError(f"Existing wheel disagrees with official pin: {filename}")
        print(json.dumps({"status": "verified_existing", "file": filename}), flush=True)
        return
    part_dir = segments_root / filename
    check_file_path(part_dir)
    part_dir.mkdir(exist_ok=True)
    ranges = [(i, start, min(start + PART_BYTES, pin["bytes"]) - 1)
              for i, start in enumerate(range(0, pin["bytes"], PART_BYTES))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch_part, url, part_dir, i, start, end, pin["bytes"])
                for i, start, end in ranges]
        for job in concurrent.futures.as_completed(jobs):
            index, count = job.result()
            print(json.dumps({"file": filename, "part": index + 1,
                              "parts": len(ranges), "bytes": count}), flush=True)
    assembled = dest / (filename + ".assembling")
    check_file_path(assembled)
    assembled.unlink(missing_ok=True)
    with assembled.open("xb") as output:
        for index, start, end in ranges:
            part = part_dir / f"part_{index:04d}"
            check_file_path(part)
            if not part.is_file() or part.stat().st_size != end - start + 1:
                raise ValueError(f"Segment missing or wrong length: {filename} {index}")
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if assembled.stat().st_size != pin["bytes"] or sha256(assembled) != pin["sha256"]:
        raise ValueError(f"Assembled wheel differs from official SHA: {filename}")
    check_file_path(final)
    if final.exists():
        raise ValueError(f"Another writer created the target wheel: {filename}")
    os.replace(assembled, final)
    print(json.dumps({"status": "verified_download", "file": filename,
                      "bytes": pin["bytes"], "sha256": pin["sha256"]}), flush=True)
    for index, _, _ in ranges:
        (part_dir / f"part_{index:04d}").unlink()
    part_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", metavar="DISTRIBUTION",
                        help="Repeat for a pinned distribution; default: all 15")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel Range requests per wheel (1–16)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    pins = load_pins()
    selected = set(args.only or ALLOWED)
    if selected - ALLOWED or (args.only and len(selected) != len(args.only)):
        raise ValueError("--only must name each approved distribution at most once")
    if (ROOT / "outputs").is_symlink() or DEST.is_symlink():
        raise ValueError("E39 wheelhouse path must not be a symlink")
    output_root = (ROOT / "outputs").resolve()
    dest = DEST.resolve()
    if output_root not in dest.parents:
        raise ValueError("Wheel destination left Git-ignored outputs/")
    dest.mkdir(parents=True, exist_ok=True)
    segments_root = dest / ".e39_pypi_segments_v1"
    check_file_path(segments_root)
    segments_root.mkdir(exist_ok=True)
    lock = dest / ".fetch_e39_pypi_cuda_wheels_v1.lock"
    check_file_path(lock)
    with lock.open("x", encoding="ascii") as handle:
        handle.write(str(os.getpid()) + "\n")
    try:
        for pin in pins:
            if pin["distribution"] in selected:
                fetch_wheel(pin, dest, segments_root, args.workers)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
