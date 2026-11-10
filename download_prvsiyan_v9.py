#!/usr/bin/env python3
"""Download the fixed public v9 files needed for a provenance audit.

The downloaded corpus stays under the Git-ignored data/external directory.
No Kaggle credentials are required for these public files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


DATASET = "prvsiyan/prvsiyans-extended-name-that-llm-dataset"
VERSION = 9
FILES = (
    "passages.csv",
    "prompts.csv",
    "provenance_summary.json",
    "provenance.jsonl",
)
OUTPUT = Path("data/external/prvsiyan")


def download_file(name: str) -> dict:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{name}"
    target = OUTPUT / name
    partial = OUTPUT / (name + ".partial")
    params = {"datasetVersionNumber": VERSION}
    with requests.get(url, params=params, headers={"Range": "bytes=0-0"}, timeout=60) as probe:
        probe.raise_for_status()
        if probe.status_code != 206:
            raise ValueError(f"{name}: server did not honor the byte-range request")
        match = re.fullmatch(r"bytes 0-0/(\d+)", probe.headers["content-range"])
        if not match:
            raise ValueError(f"{name}: unexpected Content-Range header")
        size = int(match.group(1))

    chunk_size = 1024 * 1024
    spans = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
    with partial.open("wb") as stream:
        stream.truncate(size)

    def fetch_span(span: tuple[int, int]) -> None:
        start, end = span
        for attempt in range(3):
            try:
                with requests.get(
                    url, params=params, headers={"Range": f"bytes={start}-{end}"},
                    timeout=(20, 40),
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError(f"{name}: missing 206 response for {start}-{end}")
                    payload = response.content
                    if len(payload) != end - start + 1:
                        raise ValueError(f"{name}: short range {start}-{end}")
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
        with partial.open("r+b") as stream:
            stream.seek(start)
            stream.write(payload)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch_span, spans))
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    os.replace(partial, target)
    result = {"file": name, "bytes": size, "sha256": digest.hexdigest()}
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [download_file(name) for name in FILES]
    manifest = {"dataset": DATASET, "version": VERSION, "files": files}
    (OUTPUT / "download_manifest_v9.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
