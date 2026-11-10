#!/usr/bin/env python3
"""Resume the hash-pinned official ModelScope mirror of E28's Qwen3-8B model."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from run_e28_qwen8b_lora_feasibility import EXPECTED_SHA256, MODEL_REVISION


MODEL_ID = "Qwen/Qwen3-8B"
API_ROOT = f"https://modelscope.cn/api/v1/models/{MODEL_ID}/repo"
CHUNK = 4 * 1024 * 1024
RUNTIME_EXTRA = ("generation_config.json", "vocab.json", "merges.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_mirror_manifest() -> dict[str, dict[str, int | str]]:
    hf_url = f"https://huggingface.co/api/models/{MODEL_ID}/tree/{MODEL_REVISION}?recursive=true"
    ms_url = API_ROOT + "/files?" + urllib.parse.urlencode({"Revision": "master", "Recursive": "true"})
    req_hf = urllib.request.Request(hf_url, headers={"User-Agent": "e28-pinned-mirror/1.0"})
    req_ms = urllib.request.Request(ms_url, headers={"User-Agent": "e28-pinned-mirror/1.0"})
    with urllib.request.urlopen(req_hf, timeout=45) as response:
        hf = {x["path"]: x for x in json.load(response) if x.get("type") == "file"}
    with urllib.request.urlopen(req_ms, timeout=45) as response:
        ms_data = json.load(response)
    if not ms_data.get("Success"):
        raise RuntimeError(f"ModelScope manifest failed: {ms_data.get('Message')}")
    ms = {x["Path"]: x for x in ms_data["Data"]["Files"] if x.get("Type") == "blob"}
    if not set(EXPECTED_SHA256) <= set(hf) & set(ms):
        raise ValueError("The pinned snapshot's required files are missing from the mirror")
    manifest = {}
    for name in (*EXPECTED_SHA256, *RUNTIME_EXTRA):
        item = hf.get(name)
        if item is None:
            raise ValueError(f"Pinned Hugging Face snapshot lacks runtime file: {name}")
        if name not in ms:
            raise ValueError(f"Official mirror lacks pinned file: {name}")
        other = ms[name]
        if item["size"] != other["Size"]:
            raise ValueError(f"Official mirror size differs: {name}")
        expected = other["Sha256"]
        if item.get("lfs") and item["lfs"]["oid"] != expected:
            raise ValueError(f"Official mirror LFS SHA256 differs: {name}")
        if name in EXPECTED_SHA256 and EXPECTED_SHA256[name] != expected:
            raise ValueError(f"Locked required SHA256 differs: {name}")
        manifest[name] = {"size": int(item["size"]), "sha256": expected}
    return manifest


def download_one(args: tuple[Path, str, dict[str, int | str]]) -> tuple[str, int, str]:
    directory, name, item = args
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    expected_sha = str(item["sha256"])
    if target.is_file() and target.stat().st_size == expected_size and sha256(target) == expected_sha:
        print("SKIP_VERIFIED", name, expected_size, flush=True)
        return name, expected_size, expected_sha
    part = target.with_name(target.name + ".part")
    if part.exists() and part.stat().st_size > expected_size:
        raise ValueError(f"Partial file exceeds official size: {part}")
    url = API_ROOT + "?" + urllib.parse.urlencode({"Revision": "master", "FilePath": name})
    for attempt in range(1, 11):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected_size:
            break
        headers = {"User-Agent": "e28-pinned-mirror/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Invalid resume response for {name}: {response.status} {content_range}")
                elif response.status not in (200, 206):
                    raise ValueError(f"Invalid initial response for {name}: {response.status}")
                with part.open("ab") as handle:
                    last_log = time.monotonic()
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        if time.monotonic() - last_log >= 120:
                            print("PROGRESS", name, handle.tell(), expected_size, flush=True)
                            last_log = time.monotonic()
            if part.stat().st_size == expected_size:
                break
            print("SHORT_READ", name, part.stat().st_size, expected_size, flush=True)
        except (OSError, TimeoutError) as exc:
            print("RETRY", name, attempt, type(exc).__name__, str(exc), flush=True)
            if attempt == 10:
                raise
            time.sleep(min(2 * attempt, 20))
    if not part.exists() or part.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download after retry budget: {name}")
    actual_sha = sha256(part)
    if actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for {name}: {actual_sha} != {expected_sha}")
    os.replace(part, target)
    print("VERIFIED", name, expected_size, actual_sha, flush=True)
    return name, expected_size, actual_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/qwen3-8b-b968826d"))
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()
    manifest = official_mirror_manifest()
    required_bytes = sum(int(x["size"]) for x in manifest.values())
    args.model_dir.mkdir(parents=True, exist_ok=True)
    remaining_bytes = 0
    for name, item in manifest.items():
        target = args.model_dir / name
        partial = target.with_name(target.name + ".part")
        size = int(item["size"])
        digest = str(item["sha256"])
        if target.is_file():
            if target.stat().st_size != size or sha256(target) != digest:
                raise ValueError(f"Existing final file differs from pinned official bytes: {target}")
            if partial.exists():
                raise ValueError(f"Both final and partial files exist for {name}")
        elif partial.exists():
            if partial.stat().st_size > size:
                raise ValueError(f"Partial file exceeds official size: {partial}")
            remaining_bytes += size - partial.stat().st_size
        else:
            remaining_bytes += size
    free = shutil.disk_usage(args.model_dir).free
    print("START", "revision", MODEL_REVISION, "files", len(manifest), "bytes", required_bytes,
          "bytes_remaining", remaining_bytes, "disk_free", free, flush=True)
    if free < remaining_bytes + 10 * 1024**3:
        raise OSError("Insufficient free space for remaining pinned model bytes plus 10 GiB reserve")
    started = time.monotonic()
    jobs = [(args.model_dir, name, item) for name, item in manifest.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = list(pool.map(download_one, jobs))
    report = {
        "model_id": MODEL_ID,
        "hf_revision": MODEL_REVISION,
        "mirror": "https://modelscope.cn/models/Qwen/Qwen3-8B",
        "model_dir": str(args.model_dir),
        "files": {name: {"size": size, "sha256": digest} for name, size, digest in results},
        "elapsed_seconds": time.monotonic() - started,
        "all_sha256_verified": True,
    }
    (args.model_dir / "e28_mirror_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", json.dumps({"files": len(results), "elapsed_seconds": round(report["elapsed_seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
