#!/usr/bin/env python3
"""Compare aligned OOF predictions with a paired prompt bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_e0 import LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference, dtype=str)
    candidate = pd.read_csv(args.candidate, dtype=str)
    keys = ["id", "prompt_id", "true_model", "fold"]
    required = set(keys + ["pred_model"])
    if not required.issubset(reference) or not required.issubset(candidate):
        raise ValueError("Both files must contain ID, prompt, label, fold, and prediction")
    if len(reference) != 2160 or reference["prompt_id"].nunique() != 120:
        raise ValueError("Expected the saved 2,160-row, 120-prompt development partition")
    if not reference[keys].equals(candidate[keys]):
        raise ValueError("OOF files must have identical row IDs, prompts, labels, and folds")
    if reference["id"].duplicated().any():
        raise ValueError("Duplicate row IDs")
    prompt_labels = reference.groupby("prompt_id")["true_model"].agg(list)
    if any(len(labels) != len(LABELS) or set(labels) != set(LABELS)
           for labels in prompt_labels):
        raise ValueError("Each prompt must have exactly one row for every competition label")

    ref_correct = reference["pred_model"].eq(reference["true_model"])
    cand_correct = candidate["pred_model"].eq(candidate["true_model"])
    prompt_delta = (cand_correct.astype(int) - ref_correct.astype(int)).groupby(
        reference["prompt_id"]
    ).mean().to_numpy()
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, len(prompt_delta), size=(args.repeats, len(prompt_delta)))
    deltas = prompt_delta[samples].mean(axis=1)
    ref_ba = balanced_accuracy_score(reference["true_model"], reference["pred_model"])
    cand_ba = balanced_accuracy_score(candidate["true_model"], candidate["pred_model"])
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_balanced_accuracy": float(ref_ba),
        "candidate_balanced_accuracy": float(cand_ba),
        "difference": float(cand_ba - ref_ba),
        "prompt_bootstrap_95_percentile_interval": [
            float(value) for value in np.percentile(deltas, [2.5, 97.5])
        ],
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
        "n_prompts": len(prompt_delta),
        "note": "Exploratory interval; it does not account for selecting among multiple candidate models.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
