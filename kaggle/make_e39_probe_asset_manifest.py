#!/usr/bin/env python3
"""Write a hash-only fold-1 E39 deployment manifest from ignored audit records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_e0 import LABELS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=Path("outputs/e39_recovery1_audit_snapshot"))
    parser.add_argument("--output", type=Path,
                        default=Path("kaggle/e39_probe_asset_manifest.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = json.loads((args.snapshot / "model/e39_mirror_manifest.json").read_text(encoding="utf-8"))
    fold = json.loads((args.snapshot / "fold_1/import_manifest.json").read_text(encoding="utf-8"))
    if (model.get("model_id") != "Qwen/Qwen3-8B-Base"
            or model.get("hf_revision") != "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
            or model.get("all_sha256_verified") is not True
            or fold.get("fold") != 1 or len(model.get("files", {})) != 12):
        raise ValueError("Ignored E39 snapshot is not the pinned fold-1 model")
    adapter = {name.removeprefix("adapter/"): digest
               for name, digest in fold["files"].items() if name.startswith("adapter/")}
    if (len(adapter) < 5 or len(adapter) != len(set(adapter))
            or adapter.get("adapter_model.safetensors") !=
            "576ac1ff2a26c0bed24a56db90a47fc4e13f7e8596426e95cef7dd5eb8046354"):
        raise ValueError("Fold-1 adapter hashes are incomplete")
    document = {
        "format_version": 1,
        "experiment": "e39_qwen3_8b_base_official_only_fivefold",
        "fold": 1,
        "model_repo": model["model_id"],
        "model_revision": model["hf_revision"],
        "train_sha256": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
        "labels": list(LABELS),
        "model_files": model["files"],
        "adapter_files": adapter,
        "purpose": "unscored_kaggle_t4_deployment_probe_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write bytes explicitly: Windows text-mode newline conversion would make
    # the server reference bind a different raw manifest hash from Kaggle's
    # LF-staged Notebook copy.
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(payload)
    print("E39_PROBE_MANIFEST_READY", args.output)


if __name__ == "__main__":
    main()
