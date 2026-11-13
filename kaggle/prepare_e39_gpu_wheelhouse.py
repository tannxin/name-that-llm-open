#!/usr/bin/env python3
"""Build and verify a CPython 3.13 Linux CUDA 11.8 wheelhouse for E39.

This is a local preparation tool. It never opens Kaggle, a model, or competition
data. The wheelhouse and pip logs live below Git-ignored outputs/. A completed
manifest records every wheel byte size and SHA-256; --verify uses no network.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import email.parser
import hashlib
import html.parser
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
except ImportError:  # pip itself vendors packaging on the preparation machine.
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import compatible_tags, cpython_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "kaggle/e39_gpu_wheelhouse_spec.json"
DEFAULT_DEST = ROOT / "outputs/e39_gpu_wheelhouse_cp313_cu118"
MANIFEST_NAME = "e39_gpu_wheel_manifest.json"
LOCKED_NAME = "requirements-e39-gpu-locked.txt"
TOP_LEVEL_NAME = "requirements-e39-gpu-top-level.txt"
CONSTRAINTS_NAME = "constraints-e39-reused-cp313.txt"
REPORT_NAME = "offline_resolve_report.json"
PIP_LOG_NAME = "pip_download.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_fixed(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Existing preparation file differs: {path.name}")
        return
    path.write_bytes(value)


def wheel_id(filename: str) -> tuple[str, str]:
    name, version, _, _ = parse_wheel_filename(filename)
    return canonicalize_name(name), str(version)


def wheel_metadata(path: Path) -> tuple[str, str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist()
                   if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
        if len(matches) != 1:
            raise ValueError(f"Wheel metadata count differs: {path.name}")
        stem = matches[0].split("/", 1)[0].removesuffix(".dist-info")
        if "-" not in stem:
            raise ValueError(f"Wheel top-level dist-info name malformed: {path.name}")
        stem_name, stem_version = stem.rsplit("-", 1)
        if (canonicalize_name(stem_name), stem_version) != wheel_id(path.name):
            raise ValueError(f"Wheel top-level dist-info identity differs: {path.name}")
        message = email.parser.Parser().parsestr(
            archive.read(matches[0]).decode("utf-8", errors="strict"))
    name = canonicalize_name(message["Name"])
    version = message["Version"]
    if (name, version) != wheel_id(path.name):
        raise ValueError(f"Wheel METADATA identity differs: {path.name}")
    return name, version, sorted(message.get_all("Requires-Dist", []))


def target_tag_set(spec: dict) -> set:
    platforms = spec["target"]["platform"]
    version = (3, 13)
    return (set(cpython_tags(python_version=version, abis=spec["target"]["abi"],
                             platforms=platforms)) |
            set(compatible_tags(python_version=version, interpreter="cp313",
                                platforms=platforms)))


def load_spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    target = spec["target"]
    if (spec.get("schema_version") != 1 or target.get("python") != "3.13"
            or target.get("implementation") != "cp"
            or "linux_x86_64" not in target.get("platform", [])
            or spec["requirements"][0] != "torch==2.6.0+cu118"):
        raise ValueError("E39 GPU wheelhouse spec is not the approved cp313/cu118 target")
    return spec


def linux_marker_environment() -> dict[str, str]:
    environment = default_environment()
    environment.update({
        "implementation_name": "cpython", "implementation_version": "3.13.16",
        "os_name": "posix", "platform_machine": "x86_64", "platform_python_implementation": "CPython",
        "platform_release": "", "platform_system": "Linux", "platform_version": "",
        "python_full_version": "3.13.16", "python_version": "3.13", "sys_platform": "linux",
        "extra": "",
    })
    return environment


def torch_linux_pins(spec: dict, dest: Path) -> list[str]:
    torch_path = dest / spec["primary_torch_wheel"]["file"]
    if (not torch_path.is_file() or torch_path.stat().st_size != spec["primary_torch_wheel"]["bytes"]
            or sha256(torch_path) != spec["expected_primary_wheel_sha256"][torch_path.name]):
        raise ValueError("Exact official CUDA Torch wheel must be present before Linux pin preparation")
    _, _, dependencies = wheel_metadata(torch_path)
    environment = linux_marker_environment()
    pins = []
    for line in dependencies:
        req = Requirement(line)
        name = canonicalize_name(req.name)
        if name.startswith("nvidia-") or name == "triton":
            if (req.marker is None or not req.marker.evaluate(environment=environment)
                    or len(req.specifier) != 1):
                raise ValueError(f"Unexpected Linux Torch dependency marker/specifier: {line}")
            pin = str(req.specifier)
            if not pin.startswith("==") or "*" in pin:
                raise ValueError(f"Linux Torch dependency is not an exact pin: {line}")
            pins.append(f"{name}{pin}")
    if sorted(pins) != sorted(spec["linux_torch_dependencies"]) or len(pins) != 12:
        raise ValueError("Spec Linux CUDA/Triton pins differ from official Torch wheel METADATA")
    return sorted(pins)


def linux_extra_pins(spec: dict, dest: Path) -> list[str]:
    """Lock target Linux marker deps omitted by Windows-host pip evaluation."""
    target = dest / "huggingface_hub-0.36.2-py3-none-any.whl"
    name, version, dependencies = wheel_metadata(target)
    if (name, version) != ("huggingface-hub", "0.36.2"):
        raise ValueError("Pinned Hugging Face Hub wheel differs")
    environment = linux_marker_environment()
    active = [Requirement(line) for line in dependencies
              if Requirement(line).marker is not None and
              Requirement(line).marker.evaluate(environment=environment) and
              canonicalize_name(Requirement(line).name) == "hf-xet"]
    pins = spec["linux_extra_dependencies"]
    if len(active) != 1 or pins != ["hf-xet==1.1.10"]:
        raise ValueError("Target Linux hf-xet marker/pin differs from reviewed lock")
    req = active[0]
    if not req.specifier.contains("1.1.10", prereleases=True):
        raise ValueError("Pinned hf-xet no longer satisfies Hugging Face Hub")
    return pins


def verify_target_linux_closure(selected: dict, dest: Path) -> None:
    """Audit every selected wheel's Requires-Dist against Linux CPython 3.13."""
    environment = linux_marker_environment()
    for name, item in selected.items():
        _, _, dependencies = wheel_metadata(dest / item["file"])
        for line in dependencies:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate(environment=environment):
                continue
            dependency = canonicalize_name(req.name)
            chosen = selected.get(dependency)
            if chosen is None or (req.url is not None or not req.specifier.contains(
                    chosen["version"], prereleases=True)):
                raise ValueError(f"Target Linux cp313 dependency closure fails: {name} -> {line}")


def check_dest(dest: Path) -> Path:
    root = (ROOT / "outputs").resolve()
    result = dest.resolve()
    if result != root and root not in result.parents:
        raise ValueError("Wheelhouse destination must remain below Git-ignored outputs/")
    result.mkdir(parents=True, exist_ok=True)
    return result


def seed_reused_wheels(spec: dict, dest: Path) -> list[dict]:
    source = ROOT / spec["reused_cpu_wheelhouse"]["path"]
    if sha256(source) != spec["reused_cpu_wheelhouse"]["sha256"]:
        raise ValueError("Existing CPU wheelhouse ZIP hash changed")
    with zipfile.ZipFile(source) as archive:
        old_manifest = json.loads(archive.read("manifest.json"))
        actual = {name for name in archive.namelist() if name.endswith(".whl")}
        expected = {entry["file"] for entry in old_manifest}
        if actual != expected or len(actual) != len(old_manifest):
            raise ValueError("Existing CPU wheelhouse archive entries differ from manifest")
        reused = []
        for entry in old_manifest:
            filename = entry["file"]
            if Path(filename).name != filename or not filename.endswith(".whl"):
                raise ValueError("Unsafe old wheel filename")
            name, _ = wheel_id(filename)
            if name == spec["reused_cpu_wheelhouse"]["excluded_distribution"]:
                continue
            target = dest / filename
            if target.exists():
                if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
                    raise ValueError(f"Reused wheel changed: {filename}")
            else:
                temporary = target.with_name(target.name + ".partial")
                if temporary.exists():
                    raise ValueError(f"Partial reused wheel exists: {temporary.name}")
                with archive.open(filename) as read_handle, temporary.open("xb") as write_handle:
                    for block in iter(lambda: read_handle.read(8 * 1024 * 1024), b""):
                        write_handle.write(block)
                if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"Reused wheel ZIP integrity failed: {filename}")
                os.replace(temporary, target)
            reused.append({"file": filename, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    return reused


def verify_reused_wheels(spec: dict, dest: Path) -> None:
    """Compare the current wheel bytes to the original, pinned CPU ZIP manifest."""
    source = ROOT / spec["reused_cpu_wheelhouse"]["path"]
    if sha256(source) != spec["reused_cpu_wheelhouse"]["sha256"]:
        raise ValueError("Reused CPU wheelhouse ZIP changed")
    with zipfile.ZipFile(source) as archive:
        old_manifest = json.loads(archive.read("manifest.json"))
    for entry in old_manifest:
        filename = entry["file"]
        name, _ = wheel_id(filename)
        if name == spec["reused_cpu_wheelhouse"]["excluded_distribution"]:
            continue
        target = dest / filename
        if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
            raise ValueError(f"Reused CPU wheel bytes changed after download: {filename}")


class LinkReader(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def official_torch_link(spec: dict) -> tuple[str | None, str | None]:
    """Read the primary simple index; a missing hash remains an explicit audit gap."""
    expected_name = spec["primary_torch_wheel"]["file"]
    index = spec["primary_torch_wheel"]["index"]
    try:
        with urllib.request.urlopen(index, timeout=30) as response:
            final_index = urllib.parse.urlsplit(response.geturl())
            if (response.status != 200 or final_index.scheme != "https" or
                    final_index.hostname != "download.pytorch.org"):
                raise ValueError("Official PyTorch index redirected from HTTPS primary host")
            payload = response.read(2 * 1024 * 1024).decode("utf-8")
        parser = LinkReader()
        parser.feed(payload)
        matches = [urllib.parse.urljoin(index, href) for href in parser.hrefs
                   if Path(urllib.parse.unquote(urllib.parse.urlsplit(href).path)).name == expected_name]
        if len(matches) != 1:
            return None, None
        parts = urllib.parse.urlsplit(matches[0])
        if parts.scheme != "https" or parts.hostname not in {
                "download.pytorch.org", "download-r2.pytorch.org"}:
            raise ValueError("Official PyTorch wheel link left HTTPS primary host")
        fragments = urllib.parse.parse_qs(parts.fragment)
        digest = fragments.get("sha256", [None])[0]
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Primary Torch index supplied a malformed SHA-256")
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")), digest
    except (OSError, UnicodeError) as error:
        print(f"Primary Torch index metadata unavailable: {type(error).__name__}", file=sys.stderr)
        return None, None


def official_pypi_release(name: str, version: str) -> dict[str, dict]:
    """Fetch SHA/size from PyPI itself, independent of the transfer mirror."""
    url = ("https://pypi.org/pypi/" + urllib.parse.quote(name, safe="") + "/"
           + urllib.parse.quote(version, safe="") + "/json")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                final_url = urllib.parse.urlsplit(response.geturl())
                if response.status != 200 or final_url.scheme != "https" or final_url.hostname != "pypi.org":
                    raise ValueError(f"Official PyPI URL redirected away from pypi.org: {name}")
                payload = response.read(16 * 1024 * 1024 + 1)
            if len(payload) > 16 * 1024 * 1024:
                raise ValueError(f"Official PyPI release JSON too large: {name}")
            release = json.loads(payload)
            if (canonicalize_name(release["info"]["name"]) != name or
                    release["info"]["version"] != version):
                raise ValueError(f"Official PyPI release identity differs: {name}=={version}")
            return {item["filename"]: item for item in release["urls"]}
        except (OSError, TimeoutError) as error:
            last_error = error
            time.sleep(attempt + 1)
    raise RuntimeError(f"Official PyPI release metadata unavailable: {name}=={version}") from last_error


def target_flags(spec: dict) -> list[str]:
    target = spec["target"]
    flags = ["--python-version", target["python"], "--implementation", target["implementation"]]
    for abi in target["abi"]:
        flags.extend(("--abi", abi))
    for platform in target["platform"]:
        flags.extend(("--platform", platform))
    return flags


def prepare_requirements(spec: dict, reused: list[dict], dest: Path) -> None:
    top_level = "\n".join(spec["requirements"] + torch_linux_pins(spec, dest) +
                          linux_extra_pins(spec, dest)) + "\n"
    write_fixed(dest / TOP_LEVEL_NAME, top_level.encode("ascii"))
    constraints = sorted(f"{name}=={version}" for name, version in
                         (wheel_id(entry["file"]) for entry in reused))
    if any(line.startswith("torch==") for line in constraints):
        raise ValueError("CPU Torch leaked into GPU constraints")
    write_fixed(dest / CONSTRAINTS_NAME, ("\n".join(constraints) + "\n").encode("ascii"))


def run_command(command: list[str], log: Path, *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with log.open(mode, encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                 check=False, text=True)
        handle.write(f"EXIT {process.returncode}\n")
    if process.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Offline wheel preparation command failed ({process.returncode}):\n{tail}")


def wheel_list(dest: Path) -> list[Path]:
    wheels = sorted(dest.glob("*.whl"), key=lambda path: path.name)
    identities: dict[str, str] = {}
    for path in wheels:
        name, version = wheel_id(path.name)
        if name in identities:
            raise ValueError(f"Ambiguous duplicate wheel distribution: {name}")
        identities[name] = version
        if name == "torch" and version != "2.6.0+cu118":
            raise ValueError("A non-CUDA or unexpected Torch wheel entered the GPU wheelhouse")
    return wheels


def check_wheel_tags(spec: dict, wheels: list[Path]) -> None:
    supported = target_tag_set(spec)
    for path in wheels:
        _, _, _, tags = parse_wheel_filename(path.name)
        if not tags.intersection(supported):
            raise ValueError(f"Wheel has no compatible cp313 Linux tag: {path.name}")


def url_for_download(log: str, filename: str) -> str | None:
    # Pip's public-index verbose log may expose the exact mirror URL. Hashes,
    # not this convenience field, are the integrity authority.
    pattern = re.compile(r"https://[^\s\"']+?\.whl(?:#[^\s\"']+)?")
    for candidate in pattern.findall(log):
        clean = candidate.rstrip(",)")
        if urllib.parse.unquote(urllib.parse.urlsplit(clean).path).endswith("/" + filename):
            return clean
    return None


def finish_manifest(spec: dict, reused: list[dict], dest: Path) -> dict:
    report = json.loads((dest / REPORT_NAME).read_text(encoding="utf-8"))
    selected = {}
    for item in report.get("install", []):
        metadata = item["metadata"]
        name = canonicalize_name(metadata["name"])
        version = metadata["version"]
        filename = urllib.parse.unquote(
            Path(urllib.parse.urlsplit(item["download_info"]["url"]).path).name)
        if wheel_id(filename) != (name, version):
            raise ValueError(f"Pip metadata differs from selected wheel identity: {filename}")
        if filename not in {path.name for path in dest.glob("*.whl")}:
            raise ValueError(f"Offline resolver selected a missing wheel: {filename}")
        if name in selected:
            raise ValueError(f"Offline resolver selected duplicate distribution: {name}")
        selected[name] = {"version": version, "file": filename}
    if selected.get("torch", {}).get("version") != "2.6.0+cu118":
        raise ValueError("Offline resolver did not select CUDA 11.8 Torch")
    for line in spec["requirements"]:
        name, version = line.split("==", 1)
        if selected.get(canonicalize_name(name), {}).get("version") != version:
            raise ValueError(f"Offline resolver changed direct requirement: {line}")
    for line in torch_linux_pins(spec, dest) + linux_extra_pins(spec, dest):
        name, version = line.split("==", 1)
        if selected.get(name, {}).get("version") != version:
            raise ValueError(f"Offline resolver omitted Linux Torch dependency: {line}")
    verify_target_linux_closure(selected, dest)
    reused_names = {item["file"] for item in reused}
    pip_log = (dest / PIP_LOG_NAME).read_text(encoding="utf-8", errors="replace")
    wheels = []
    all_wheels = wheel_list(dest)
    check_wheel_tags(spec, all_wheels)
    identities = {wheel_id(path.name) for path in all_wheels
                  if path.name not in reused_names and wheel_id(path.name)[0] != "torch"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        release_metadata = dict(zip(sorted(identities), pool.map(
            lambda pair: official_pypi_release(*pair), sorted(identities))))
    for path in all_wheels:
        name, version = wheel_id(path.name)
        _, _, dependencies = wheel_metadata(path)
        entry = {"file": path.name, "bytes": path.stat().st_size,
                 "sha256": sha256(path), "distribution": name, "version": version,
                 "selected": name in selected,
                 "source": "verified_cpu_wheelhouse" if path.name in reused_names else "public_index"}
        if name in selected:
            entry["requires_dist"] = dependencies
        if path.name not in reused_names:
            entry["origin_url"] = url_for_download(pip_log, path.name)
        if name != "torch" and path.name not in reused_names:
            official = release_metadata[(name, version)].get(path.name)
            if official is None or official["digests"]["sha256"] != entry["sha256"] or \
                    official["size"] != entry["bytes"]:
                raise ValueError(f"Official PyPI SHA-256/size mismatch: {path.name}")
            entry["official_pypi_sha256"] = official["digests"]["sha256"]
            entry["official_pypi_bytes"] = official["size"]
        wheels.append(entry)
    for filename, expected in spec["expected_primary_wheel_sha256"].items():
        match = next((row for row in wheels if row["file"] == filename), None)
        if match is None or match["sha256"] != expected:
            raise ValueError(f"Official primary wheel SHA-256 mismatch: {filename}")
    torch_file = spec["primary_torch_wheel"]["file"]
    torch_entry = next((row for row in wheels if row["file"] == torch_file), None)
    if torch_entry is None or not torch_entry["selected"]:
        raise ValueError("Primary CPython 3.13 CUDA 11.8 Torch wheel was not selected")
    torch_url, official_sha = official_torch_link(spec)
    if official_sha is not None and torch_entry["sha256"] != official_sha:
        raise ValueError("Primary Torch index SHA-256 differs from downloaded wheel")
    locked = "\n".join(
        f"{name}=={item['version']} --hash=sha256:{next(row['sha256'] for row in wheels if row['file'] == item['file'])}"
        for name, item in sorted(selected.items())) + "\n"
    write_fixed(dest / LOCKED_NAME, locked.encode("ascii"))
    manifest = {
        "schema_version": 1,
        "purpose": spec["purpose"],
        "target": spec["target"],
        "spec_sha256": sha256(SPEC_PATH),
        "reused_cpu_wheelhouse_sha256": spec["reused_cpu_wheelhouse"]["sha256"],
        "requirements_sha256": sha256(dest / TOP_LEVEL_NAME),
        "constraints_sha256": sha256(dest / CONSTRAINTS_NAME),
        "locked_requirements_sha256": sha256(dest / LOCKED_NAME),
        "selected_packages": selected,
        "primary_torch_origin_url": torch_url,
        "primary_torch_index_sha256": official_sha,
        "primary_torch_index_hash_verified": official_sha is not None,
        "wheels": wheels,
    }
    write_fixed(dest / MANIFEST_NAME,
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def verify(spec: dict, dest: Path) -> dict:
    path = dest / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError("Completed E39 GPU wheel manifest is absent")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("spec_sha256") != sha256(SPEC_PATH):
        raise ValueError("E39 GPU manifest spec hash/schema changed")
    if manifest.get("selected_packages", {}).get("torch", {}).get("version") != "2.6.0+cu118":
        raise ValueError("E39 GPU manifest does not select exact CUDA Torch")
    if manifest["selected_packages"]["torch"]["file"] != spec["primary_torch_wheel"]["file"]:
        raise ValueError("E39 GPU selected Torch wheel filename differs from primary index")
    if manifest.get("primary_torch_index_sha256") is not None:
        torch_entry = next((row for row in manifest["wheels"]
                            if row["file"] == spec["primary_torch_wheel"]["file"]), None)
        if torch_entry is None or torch_entry["sha256"] != manifest["primary_torch_index_sha256"]:
            raise ValueError("Primary Torch index hash no longer matches manifest")
    for filename, key in ((TOP_LEVEL_NAME, "requirements_sha256"),
                          (CONSTRAINTS_NAME, "constraints_sha256"),
                          (LOCKED_NAME, "locked_requirements_sha256")):
        if sha256(dest / filename) != manifest[key]:
            raise ValueError(f"E39 GPU {filename} hash changed")
    entries = manifest.get("wheels", [])
    actual = {path.name for path in dest.glob("*.whl")}
    expected = {entry["file"] for entry in entries}
    if actual != expected or len(entries) != len(expected):
        raise ValueError("E39 GPU wheel list changed")
    with zipfile.ZipFile(ROOT / spec["reused_cpu_wheelhouse"]["path"]) as archive:
        old_manifest = json.loads(archive.read("manifest.json"))
    reused_files = {item["file"] for item in old_manifest
                    if wheel_id(item["file"])[0] != spec["reused_cpu_wheelhouse"]["excluded_distribution"]}
    for entry in entries:
        path = dest / entry["file"]
        if (Path(entry["file"]).name != entry["file"] or
                path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]):
            raise ValueError(f"E39 GPU wheel integrity failed: {entry['file']}")
        if wheel_id(entry["file"]) != (entry["distribution"], entry["version"]):
            raise ValueError(f"Wheel filename and locked identity differ: {entry['file']}")
        expected_source = ("verified_cpu_wheelhouse" if entry["file"] in reused_files
                           else "public_index")
        if entry["source"] != expected_source:
            raise ValueError(f"Wheel source label differs from pinned CPU ZIP: {entry['file']}")
        if entry["distribution"] != "torch" and entry["source"] != "verified_cpu_wheelhouse" and (entry.get("official_pypi_sha256") !=
                entry["sha256"] or entry.get("official_pypi_bytes") != entry["bytes"]):
            raise ValueError(f"Official PyPI digest record differs: {entry['file']}")
    all_wheels = wheel_list(dest)
    check_wheel_tags(spec, all_wheels)
    verify_reused_wheels(spec, dest)
    for filename, expected_hash in spec["expected_primary_wheel_sha256"].items():
        row = next((item for item in entries if item["file"] == filename), None)
        if row is None or row["sha256"] != expected_hash:
            raise ValueError(f"Published primary wheel digest differs: {filename}")
    selected = manifest["selected_packages"]
    for line in spec["requirements"]:
        name, version = line.split("==", 1)
        if selected.get(canonicalize_name(name), {}).get("version") != version:
            raise ValueError(f"Direct GPU requirement differs from spec: {line}")
    for line in torch_linux_pins(spec, dest) + linux_extra_pins(spec, dest):
        name, version = line.split("==", 1)
        if selected.get(name, {}).get("version") != version:
            raise ValueError(f"Pinned Linux Torch dependency differs from manifest: {line}")
    rows_by_file = {row["file"]: row for row in entries}
    for name, item in selected.items():
        row = rows_by_file.get(item["file"])
        if row is None or (row["distribution"], row["version"]) != (name, item["version"]):
            raise ValueError(f"Selected package lacks matching verified wheel: {name}")
        if not row["selected"]:
            raise ValueError(f"Selected package not marked selected: {name}")
        _, _, dependencies = wheel_metadata(dest / item["file"])
        if row.get("requires_dist") != dependencies:
            raise ValueError(f"Selected wheel dependency metadata changed: {name}")
    expected_locked = "\n".join(
        f"{name}=={item['version']} --hash=sha256:{rows_by_file[item['file']]['sha256']}"
        for name, item in sorted(selected.items())) + "\n"
    if (dest / LOCKED_NAME).read_text(encoding="ascii") != expected_locked:
        raise ValueError("Full transitive hash-locked requirement list differs from manifest")
    verify_target_linux_closure(selected, dest)
    return manifest


def refresh_primary_index(spec: dict, dest: Path) -> dict:
    """Attach live official Torch index evidence to a verified local manifest."""
    manifest = verify(spec, dest)
    url, official_sha = official_torch_link(spec)
    expected = spec["expected_primary_wheel_sha256"][spec["primary_torch_wheel"]["file"]]
    if url is None or official_sha != expected:
        raise ValueError("Primary PyTorch HTTPS index did not confirm the pinned CUDA Torch SHA")
    if manifest.get("primary_torch_index_hash_verified"):
        if manifest["primary_torch_index_sha256"] != official_sha:
            raise ValueError("Existing primary PyTorch index evidence differs")
        return manifest
    backup = ROOT / "outputs/e39_gpu_manifest_before_primary_index_refresh.json"
    write_fixed(backup, (dest / MANIFEST_NAME).read_bytes())
    manifest["primary_torch_origin_url"] = url
    manifest["primary_torch_index_sha256"] = official_sha
    manifest["primary_torch_index_hash_verified"] = True
    temporary = dest / (MANIFEST_NAME + ".refreshing")
    if temporary.exists():
        raise ValueError("Prior primary index manifest refresh artifact exists")
    temporary.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(temporary, dest / MANIFEST_NAME)
    return verify(spec, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "verify", "refresh-primary-index"))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    spec = load_spec()
    dest = check_dest(args.dest)
    if args.action == "verify":
        manifest = verify(spec, dest)
    elif args.action == "refresh-primary-index":
        manifest = refresh_primary_index(spec, dest)
    else:
        if (dest / MANIFEST_NAME).exists():
            raise ValueError("Completed manifest exists; use verify, never silently redownload")
        reused = seed_reused_wheels(spec, dest)
        prepare_requirements(spec, reused, dest)
        common = ["--only-binary=:all:", "--no-cache-dir", *target_flags(spec)]
        if len(spec["source_indexes"]) != 1:
            raise ValueError("Expected one pinned package transfer index")
        download = [sys.executable, "-m", "pip", "download", "--dest", str(dest),
                    "--find-links", str(dest), "--index-url", spec["source_indexes"][0],
                    *common, "-r", str(dest / TOP_LEVEL_NAME),
                    "-c", str(dest / CONSTRAINTS_NAME)]
        run_command(download, dest / PIP_LOG_NAME)
        verify_reused_wheels(spec, dest)
        offline = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                   "--no-index", "--find-links", str(dest), "--report", str(dest / REPORT_NAME),
                   *common, "-r", str(dest / TOP_LEVEL_NAME),
                   "-c", str(dest / CONSTRAINTS_NAME)]
        run_command(offline, dest / "pip_offline_resolution.log")
        manifest = finish_manifest(spec, reused, dest)
        verify(spec, dest)
    total = sum(item["bytes"] for item in manifest["wheels"])
    print(json.dumps({"status": "verified", "manifest_sha256": sha256(dest / MANIFEST_NAME),
                      "wheel_count": len(manifest["wheels"]), "wheel_bytes": total,
                      "selected_packages": len(manifest["selected_packages"])}))


if __name__ == "__main__":
    main()
