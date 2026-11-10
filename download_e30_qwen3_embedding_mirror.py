#!/usr/bin/env python3
"""Check or resume the byte-pinned official ModelScope Qwen3-Embedding-8B mirror.

``check`` reads metadata only. ``check --verify-hf`` additionally fetches four
small Hugging Face files (under 50 KiB total) and the pinned tree metadata.
Only the explicit ``download`` command transfers weight shards. No GPU is used.
"""

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
import uuid
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
HF_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
# Qwen-owned ModelScope repository HEAD/master resolved by git ls-remote on
# 2026-11-10. Both the files and download endpoints accept this commit SHA.
MS_REVISION = "fcd9221ff0c460fdd81bfd04f4bc5b014e58da5b"
MS_API_ROOT = f"https://modelscope.cn/api/v1/models/{MODEL_ID}/repo"
HF_BASE = f"https://huggingface.co/{MODEL_ID}"
CHUNK = 4 * 1024 * 1024
USER_AGENT = "name-that-llm-e30-pinned-mirror/1.0"
SMALL_HF_FILES = ("LICENSE", "config.json", "model.safetensors.index.json",
                  "tokenizer_config.json")

# Verified on 2026-11-10 against the exact Hugging Face commit and the Qwen-owned
# ModelScope repository. Xet/LFS SHA-256 covers the four weight files and
# tokenizer.json; the four small files were fetched byte-for-byte from HF.
EXPECTED = {
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "config.json": (729, "dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7"),
    "model.safetensors.index.json": (30432, "ceaf1429bfa18eecc75d079737e8d1fe8dc0627f5ec7e6a985514626d1ac6a77"),
    "tokenizer.json": (11422947, "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d"),
    "tokenizer_config.json": (7256, "2f58f4bbd7bbce15d683f525954ef3a92cd82f5e06415a9c513859bf8ab72436"),
    "model-00001-of-00004.safetensors": (4900037024, "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451"),
    "model-00002-of-00004.safetensors": (4915959512, "dff635b0f6dbbaad2a2d633ef037ec0a39bc165cc1806c712fbd6fcbcb4526c0"),
    "model-00003-of-00004.safetensors": (4983067656, "30b1d4c53d84eb018f642cad7b373f0aabf79699872d8702c1f38577c0a59a2f"),
    "model-00004-of-00004.safetensors": (335570376, "36cbc9c60375693629f25743c1e77ebb1724af58e671b2376463193c7fd21ef6"),
}
WEIGHT_NAMES = tuple(name for name in EXPECTED if name.endswith(".safetensors"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_url(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def official_mirror_manifest() -> dict[str, dict[str, int | str]]:
    url = MS_API_ROOT + "/files?" + urllib.parse.urlencode(
        {"Revision": MS_REVISION, "Recursive": "true"})
    payload = read_json_url(url)
    if not isinstance(payload, dict) or not payload.get("Success"):
        raise RuntimeError(f"ModelScope manifest request failed: {payload}")
    files = {item["Path"]: item for item in payload["Data"]["Files"]
             if item.get("Type") == "blob"}
    missing = sorted(set(EXPECTED) - set(files))
    if missing:
        raise ValueError(f"Official ModelScope mirror lacks pinned files: {missing}")
    result = {}
    for name, (expected_size, expected_sha) in EXPECTED.items():
        item = files[name]
        if item["Size"] != expected_size or item.get("Sha256") != expected_sha:
            raise ValueError(f"Mirror metadata changed for {name}; stop before download")
        result[name] = {"size": expected_size, "sha256": expected_sha}
    return result


def verify_hf_small_files_and_tree() -> dict[str, bool]:
    url = f"https://huggingface.co/api/models/{MODEL_ID}/tree/{HF_REVISION}?recursive=true"
    tree = read_json_url(url)
    if not isinstance(tree, list):
        raise ValueError("Hugging Face pinned tree metadata is invalid")
    files = {item["path"]: item for item in tree if item.get("type") == "file"}
    if not set(EXPECTED) <= set(files):
        raise ValueError("Hugging Face pinned tree lacks a required file")
    for name, (size, digest) in EXPECTED.items():
        item = files[name]
        if item.get("size") != size:
            raise ValueError(f"Hugging Face pinned size differs for {name}")
        if name in (*WEIGHT_NAMES, "tokenizer.json") and item.get("lfs", {}).get("oid") != digest:
            raise ValueError(f"Hugging Face pinned LFS SHA differs for {name}")
    for name in SMALL_HF_FILES:
        request = urllib.request.Request(
            f"{HF_BASE}/resolve/{HF_REVISION}/{name}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read(1_000_000)
        expected_size, expected_sha = EXPECTED[name]
        if len(body) != expected_size or hashlib.sha256(body).hexdigest() != expected_sha:
            raise ValueError(f"Hugging Face pinned small file differs for {name}")
    return {"hf_tree_sizes_and_lfs_sha_match": True, "hf_small_file_bytes_match": True}


def quarantine_partial(partial: Path, observed_sha: str) -> Path:
    # Keep the evidence instead of deleting a corrupt multi-GB partial file.
    quarantine = partial.with_name(
        f"{partial.name}.corrupt-{observed_sha[:12]}-{uuid.uuid4().hex}")
    if partial.resolve().parent != quarantine.resolve().parent or quarantine.exists():
        raise ValueError("Unsafe E30 partial-file quarantine target")
    os.replace(partial, quarantine)
    print("QUARANTINED_CORRUPT_PARTIAL", partial.name, quarantine.name, flush=True)
    return quarantine


def inspect_existing(model_dir: Path) -> tuple[int, int, int]:
    remaining = 0
    verified = 0
    quarantined = 0
    for name, (size, digest) in EXPECTED.items():
        target = model_dir / name
        partial = target.with_name(target.name + ".part")
        if target.is_file():
            if target.stat().st_size != size or sha256(target) != digest:
                raise ValueError(f"Existing final file differs from pinned official bytes: {target}")
            if partial.exists():
                raise ValueError(f"Both final and partial files exist for {name}")
            verified += 1
        elif partial.is_file():
            if partial.stat().st_size > size:
                raise ValueError(f"Partial file exceeds pinned official size: {partial}")
            if partial.stat().st_size == size:
                observed = sha256(partial)
                if observed != digest:
                    quarantine_partial(partial, observed)
                    remaining += size
                    quarantined += 1
            else:
                remaining += size - partial.stat().st_size
        else:
            remaining += size
    return remaining, verified, quarantined


def download_one(job: tuple[Path, str, dict[str, int | str]]) -> tuple[str, int, str]:
    model_dir, name, item = job
    target = model_dir / name
    size, digest = int(item["size"]), str(item["sha256"])
    if target.is_file():
        if target.stat().st_size != size or sha256(target) != digest:
            raise ValueError(f"Existing final file differs from pinned official bytes: {target}")
        print("SKIP_VERIFIED", name, size, flush=True)
        return name, size, digest
    partial = target.with_name(target.name + ".part")
    url = MS_API_ROOT + "?" + urllib.parse.urlencode(
        {"Revision": MS_REVISION, "FilePath": name})
    for attempt in range(1, 11):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file exceeds pinned official size: {partial}")
        if offset == size:
            break
        headers = {"User-Agent": USER_AGENT}
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
                with partial.open("ab") as handle:
                    last_log = time.monotonic()
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        if time.monotonic() - last_log >= 120:
                            print("PROGRESS", name, handle.tell(), size, flush=True)
                            last_log = time.monotonic()
            if partial.stat().st_size == size:
                break
            print("SHORT_READ", name, partial.stat().st_size, size, flush=True)
            if attempt < 10:
                pause = min(2 ** attempt, 30)
                print("BACKOFF_SECONDS", name, pause, flush=True)
                time.sleep(pause)
        except (OSError, TimeoutError) as error:
            print("RETRY", name, attempt, type(error).__name__, str(error), flush=True)
            if attempt == 10:
                raise
            pause = min(2 ** attempt, 30)
            print("BACKOFF_SECONDS", name, pause, flush=True)
            time.sleep(pause)
    if not partial.is_file() or partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download after retry budget: {name}")
    observed = sha256(partial)
    if observed != digest:
        quarantine_partial(partial, observed)
        raise ValueError(f"Downloaded SHA-256 mismatch for {name}; corrupt partial quarantined. "
                         "Rerun download to recheck free space and fetch this file anew")
    os.replace(partial, target)
    print("VERIFIED", name, size, observed, flush=True)
    return name, size, observed


def existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current == current.parent:
            raise OSError(f"No existing ancestor for {path}")
        current = current.parent
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    check = stages.add_parser("check", help="Metadata only; no weight transfer")
    check.add_argument("--verify-hf", action="store_true",
                       help="Also read HF pinned tree and four small files (<50 KiB)")
    download = stages.add_parser("download", help="Explicitly fetch and verify the pinned mirror")
    download.add_argument("--model-dir", type=Path,
                          default=Path("data/models/qwen3-embedding-8b-1d8ad4c"))
    download.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    manifest = official_mirror_manifest()
    total_bytes = sum(int(item["size"]) for item in manifest.values())
    if args.stage == "check":
        result = {"model_id": MODEL_ID, "hf_revision": HF_REVISION,
                  "modelscope_revision": MS_REVISION,
                  "mirror": f"https://modelscope.cn/models/{MODEL_ID}",
                  "files": len(manifest), "total_bytes": total_bytes,
                  "modelscope_metadata_matches_pinned_hf_bytes": True,
                  "weights_downloaded": False}
        if args.verify_hf:
            result.update(verify_hf_small_files_and_tree())
        print(json.dumps(result, indent=2), flush=True)
        return
    if not 1 <= args.max_workers <= 5:
        raise ValueError("--max-workers must be in 1..5")
    remaining, verified, quarantined = inspect_existing(args.model_dir)
    free = shutil.disk_usage(existing_ancestor(args.model_dir)).free
    reserve = 10 * 1024**3
    print("START", MODEL_ID, "hf_revision", HF_REVISION,
          "modelscope_revision", MS_REVISION, "total_bytes", total_bytes,
          "remaining_bytes", remaining, "already_verified", verified,
          "quarantined_corrupt_partials", quarantined,
          "disk_free", free, flush=True)
    if free < remaining + reserve:
        raise OSError("Insufficient disk space for remaining pinned files plus 10 GiB reserve")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    jobs = [(args.model_dir, name, item) for name, item in manifest.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = list(pool.map(download_one, jobs))
    report = {
        "model_id": MODEL_ID, "hf_revision": HF_REVISION,
        "modelscope_revision": MS_REVISION,
        "mirror": f"https://modelscope.cn/models/{MODEL_ID}",
        "files": {name: {"size": size, "sha256": digest}
                  for name, size, digest in results},
        "elapsed_seconds": time.monotonic() - started,
        "all_sha256_verified": True,
    }
    (args.model_dir / "e30_mirror_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", len(results), "files", flush=True)


if __name__ == "__main__":
    main()
