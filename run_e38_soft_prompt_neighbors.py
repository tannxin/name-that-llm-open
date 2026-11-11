#!/usr/bin/env python3
"""E38: locked CPU same-prompt retrieval diagnostic and gated score check.

The retrieval rule sees response text and IDs, never prompt IDs or model labels.
Prompt IDs are used only to construct deterministic sparse-batch simulations
and to audit retrieved neighbors. The preflight and diagnostic stages never
read class scores or BA; scoring is blocked until the diagnostic gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data/train.csv"
E0_DIR = ROOT / "outputs/e0_tfidf_svm"
PREREG = ROOT / "docs/e38_soft_prompt_neighbors_preregistration.md"
OUTPUT = ROOT / "outputs/e38_soft_prompt_neighbors"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
MANIFEST_SHA256 = "653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15"
OOF_SHA256 = "698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa"
FOLDS = (1, 2, 3, 4, 5)
SEED = 42
TOP_K = 5
VECTORIZER = {
    "ngram_range": (1, 2),
    "lowercase": True,
    "stop_words": "english",
    "sublinear_tf": True,
    "min_df": 2,
    "max_features": 100_000,
    "norm": "l2",
    "dtype": np.float32,
}
COSINE_THRESHOLD = 0.20
SPARSE_PANELS = (1, 2, 3, 4, 8)
MIN_SPARSE_TOP1 = 0.60
MIN_SPARSE_COVERAGE = 0.70
CENTERING_COEFFICIENT = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Read only development rows after checking the locked E0 manifest."""
    if not PREREG.is_file() or PREREG.stat().st_size == 0:
        raise ValueError("E38 preregistration is required before preflight or diagnostic")
    if sha256(TRAIN) != TRAIN_SHA256:
        raise ValueError("Official training file differs from the locked dataset")
    manifest_path = E0_DIR / "split_manifest.csv"
    oof_path = E0_DIR / "oof_predictions.csv"
    if sha256(manifest_path) != MANIFEST_SHA256 or sha256(oof_path) != OOF_SHA256:
        raise ValueError("E0 development split or OOF identity file changed")

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if list(manifest.columns) != ["prompt_id", "split"] or manifest["prompt_id"].duplicated().any():
        raise ValueError("E0 split manifest structure changed")
    if set(manifest["split"]) != {"development", "holdout"}:
        raise ValueError("E0 split labels changed")
    dev_prompts = set(manifest.loc[manifest["split"].eq("development"), "prompt_id"])
    if len(manifest) != 150 or len(dev_prompts) != 120 or manifest["split"].eq("holdout").sum() != 30:
        raise ValueError("Unexpected E0 development/holdout prompt counts")

    rows: list[dict[str, str]] = []
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"id", "prompt_id", "text", "model"}:
            raise ValueError("Official training columns changed")
        for row in reader:
            if row["prompt_id"] in dev_prompts:
                rows.append({key: row[key] for key in ("id", "prompt_id", "text", "model")})
    dev = pd.DataFrame(rows)
    oof = pd.read_csv(oof_path, dtype=str, keep_default_na=False,
                      usecols=["id", "prompt_id", "true_model", "fold"])
    expected = dev.loc[:, ["id", "prompt_id", "model"]].rename(columns={"model": "true_model"})
    if len(dev) != 2160 or dev["id"].duplicated().any() or dev["prompt_id"].nunique() != 120:
        raise ValueError("Unexpected development rows or prompt count")
    if not expected.reset_index(drop=True).equals(oof.loc[:, ["id", "prompt_id", "true_model"]]):
        raise ValueError("Official development identity does not match E0 OOF")
    if not dev.groupby("prompt_id").size().eq(18).all():
        raise ValueError("Expected 18 official responses per development prompt")
    folds = pd.to_numeric(oof["fold"], errors="raise").to_numpy(dtype=np.int64)
    if set(folds) != set(FOLDS):
        raise ValueError("Unexpected E0 fold identifiers")
    for fold in FOLDS:
        fit, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if len(fit) != 1728 or len(valid) != 432:
            raise ValueError(f"Fold {fold} has unexpected row counts")
        if set(dev.iloc[fit]["prompt_id"]) & set(dev.iloc[valid]["prompt_id"]):
            raise ValueError(f"Fold {fold} overlaps train and validation prompts")
        if dev.iloc[valid]["prompt_id"].nunique() != 24:
            raise ValueError(f"Fold {fold} has unexpected validation prompts")
        if not dev.iloc[valid].groupby("model").size().eq(24).all() or dev.iloc[valid]["model"].nunique() != 18:
            raise ValueError(f"Fold {fold} has unexpected class support")

    metadata = {
        "official_train_sha256": TRAIN_SHA256,
        "e0_split_manifest_sha256": MANIFEST_SHA256,
        "e0_oof_sha256": OOF_SHA256,
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__).resolve()),
        "n_development_rows": len(dev),
        "n_development_prompts": int(dev["prompt_id"].nunique()),
        "n_folds": len(FOLDS),
        "retrieval_uses_prompt_id": False,
        "retrieval_uses_model_label": False,
        "fit_classifiers": False,
        "read_ba_or_model_scores": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    return dev, folds, metadata


def scenario_indices(prompt_ids: np.ndarray, ids: np.ndarray, panel: int | None) -> np.ndarray:
    """Make disjoint 1+2+3+4+8 panels; retrieval itself never sees prompt IDs."""
    if panel is None:
        return np.arange(len(ids), dtype=np.int64)
    if panel not in SPARSE_PANELS:
        raise ValueError("Invalid frozen sparse panel")
    offset = sum(size for size in SPARSE_PANELS if size < panel)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, prompt_id in enumerate(prompt_ids):
        grouped[str(prompt_id)].append(i)
    selected: list[int] = []
    for positions in grouped.values():
        if len(positions) != 18:
            raise ValueError("Sparse simulation requires 18 original rows per prompt")
        ordered = sorted(positions, key=lambda i: (
            hashlib.sha256(("e38:sparse:v1:" + ids[i]).encode("utf-8")).digest(), ids[i]))
        selected.extend(ordered[offset:offset + panel])
    return np.asarray(sorted(selected), dtype=np.int64)


COUNT_KEYS = ("top1_hits", "same_slots", "qualified_slots", "covered_queries",
              "covered_eligible_queries", "oracle_slots")


def summarize_counts(counts: dict[str, int], n: int, eligible: int) -> dict:
    return {
        "n_queries": n,
        "n_eligible_queries": eligible,
        "same_prompt_top1_rate_all": counts["top1_hits"] / n,
        "same_prompt_top1_rate_eligible": counts["top1_hits"] / eligible if eligible else None,
        "same_prompt_precision_at_5": counts["same_slots"] / counts["qualified_slots"]
            if counts["qualified_slots"] else None,
        "qualified_neighbor_coverage_all": counts["covered_queries"] / n,
        "qualified_neighbor_coverage_eligible": counts["covered_eligible_queries"] / eligible
            if eligible else None,
        "same_prompt_recall_of_available_at_5": counts["same_slots"] / counts["oracle_slots"]
            if counts["oracle_slots"] else None,
        "counts": counts,
    }


def retrieve_neighbors(similarities: np.ndarray, ids: np.ndarray,
                       subset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve using text similarity and IDs; no true prompt or model labels."""
    local_scores = similarities[np.ix_(subset, subset)]
    local_ids = ids[subset]
    neighbors_out = np.full((len(subset), TOP_K), -1, dtype=np.int32)
    weights_out = np.zeros((len(subset), TOP_K), dtype=np.float32)
    for query in range(len(subset)):
        row = local_scores[query].copy()
        row[query] = -np.inf
        ranked = np.lexsort((local_ids, -row))
        qualified = ranked[row[ranked] >= COSINE_THRESHOLD][:TOP_K]
        if not len(qualified):
            continue
        neighbors_out[query, :len(qualified)] = subset[qualified]
        squared = np.square(row[qualified].astype(np.float64))
        weights_out[query, :len(qualified)] = (squared / squared.sum()).astype(np.float32)
    return neighbors_out, weights_out


def audit_neighbors(prompt_ids: np.ndarray, rows: np.ndarray,
                    neighbor_map: np.ndarray) -> dict:
    """Audit a persisted map with true prompt IDs only after retrieval ends."""
    group_size = pd.Series(prompt_ids[rows]).value_counts().to_dict()
    in_batch = np.zeros(len(prompt_ids), dtype=bool)
    in_batch[rows] = True
    counts = {key: 0 for key in COUNT_KEYS}
    eligible = 0
    for row in rows:
        neighbors = neighbor_map[row]
        neighbors = neighbors[neighbors >= 0]
        if len(neighbors) and not in_batch[neighbors].all():
            raise ValueError("A retrieved neighbor is outside its simulated batch")
        available = min(TOP_K, int(group_size[prompt_ids[row]]) - 1)
        eligible += int(available > 0)
        counts["oracle_slots"] += available
        if not len(neighbors):
            continue
        matches = prompt_ids[neighbors] == prompt_ids[row]
        counts["top1_hits"] += int(matches[0])
        counts["same_slots"] += int(matches.sum())
        counts["qualified_slots"] += len(neighbors)
        counts["covered_queries"] += 1
        counts["covered_eligible_queries"] += int(available > 0)
    return summarize_counts(counts, len(rows), eligible)


def aggregate_reports(reports: list[dict], scenario: str) -> dict:
    chosen = [row for row in reports if row["scenario"] == scenario]
    counts = {key: sum(row["counts"][key] for row in chosen) for key in COUNT_KEYS}
    n = sum(row["n_queries"] for row in chosen)
    eligible = sum(row["n_eligible_queries"] for row in chosen)
    return {"scenario": scenario, **summarize_counts(counts, n, eligible)}


def retrieve_maps(dev: pd.DataFrame, folds: np.ndarray) -> tuple[dict, dict[int, int]]:
    scenarios = ("full18", *(f"panel_{k}" for k in SPARSE_PANELS))
    maps = {name: np.full((len(dev), TOP_K), -1, dtype=np.int32)
            for name in ("full_neighbors", "sparse_neighbors")}
    maps.update({name: np.zeros((len(dev), TOP_K), dtype=np.float32)
                 for name in ("full_weights", "sparse_weights")})
    maps["sparse_panel"] = np.zeros(len(dev), dtype=np.uint8)
    sparse_assigned = np.zeros(len(dev), dtype=bool)
    vocabulary_sizes: dict[int, int] = {}
    for fold in FOLDS:
        fit = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        vectorizer = TfidfVectorizer(**VECTORIZER)
        vectorizer.fit(dev.iloc[fit]["text"].tolist())
        vocabulary_sizes[fold] = len(vectorizer.vocabulary_)
        vectors = vectorizer.transform(dev.iloc[valid]["text"].tolist())
        cosine = (vectors @ vectors.T).toarray().astype(np.float32, copy=False)
        ids = dev.iloc[valid]["id"].to_numpy(dtype=str)
        prompts = dev.iloc[valid]["prompt_id"].to_numpy(dtype=str)
        for scenario in scenarios:
            panel = None if scenario == "full18" else int(scenario[6:])
            subset = scenario_indices(prompts, ids, panel)
            neighbors, weights = retrieve_neighbors(cosine, ids, subset)
            key = "full" if panel is None else "sparse"
            global_rows = valid[subset]
            maps[f"{key}_neighbors"][global_rows] = np.where(
                neighbors >= 0, valid[np.maximum(neighbors, 0)], -1)
            maps[f"{key}_weights"][global_rows] = weights
            if panel is not None:
                if sparse_assigned[global_rows].any():
                    raise ValueError("A sparse response was assigned to multiple panels")
                sparse_assigned[global_rows] = True
                maps["sparse_panel"][global_rows] = panel
    if not sparse_assigned.all():
        raise ValueError("Sparse panels did not recompose all development rows")
    maps["ids"] = dev["id"].to_numpy(dtype=str)
    maps["folds"] = folds
    return maps, vocabulary_sizes


def audit_diagnostic(dev: pd.DataFrame, folds: np.ndarray, metadata: dict,
                     maps: dict, vocabulary_sizes: dict[int, int], map_sha256: str) -> dict:
    """Inspect true prompts only after the text-only neighbor map is persisted."""
    scenarios = ("full18", *(f"panel_{k}" for k in SPARSE_PANELS))
    reports: list[dict] = []
    prompt_ids = dev["prompt_id"].to_numpy(dtype=str)
    ids = dev["id"].to_numpy(dtype=str)
    for fold in FOLDS:
        valid = np.flatnonzero(folds == fold)
        fold_prompts = prompt_ids[valid]
        fold_ids = ids[valid]
        for scenario in scenarios:
            panel = None if scenario == "full18" else int(scenario[6:])
            rows = valid[scenario_indices(fold_prompts, fold_ids, panel)]
            key = "full_neighbors" if panel is None else "sparse_neighbors"
            reports.append({"fold": fold, "scenario": scenario,
                            **audit_neighbors(prompt_ids, rows, maps[key]),
                            "train_vocabulary_size": vocabulary_sizes[fold]})
    pooled = [aggregate_reports(reports, scenario) for scenario in scenarios]
    sparse_counts = {key: sum(row["counts"][key] for row in pooled[1:]) for key in COUNT_KEYS}
    sparse_eligible = sum(row["n_eligible_queries"] for row in pooled[1:])
    sparse_all = {"scenario": "sparse_recomposed", **summarize_counts(sparse_counts, len(dev), sparse_eligible)}
    gate = {
        "eligible_top1_at_least_0_60":
            sparse_all["same_prompt_top1_rate_eligible"] >= MIN_SPARSE_TOP1,
        "eligible_coverage_at_least_0_70":
            sparse_all["qualified_neighbor_coverage_eligible"] >= MIN_SPARSE_COVERAGE,
    }
    report = {
        "experiment": "e38_soft_prompt_neighbors_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata,
        "vectorizer": {**VECTORIZER, "dtype": "float32"},
        "top_k": TOP_K,
        "cosine_threshold": COSINE_THRESHOLD,
        "simulation": "SHA256('e38:sparse:v1:' + id) within each prompt; disjoint 1+2+3+4+8 panels",
        "folds": reports,
        "pooled": pooled,
        "sparse_recomposed": sparse_all,
        "score_stage_gate": gate,
        "score_stage_allowed": all(gate.values()),
        "contains_classifier_or_ba_scores": False,
        "neighbor_map_sha256": map_sha256,
    }
    return report


def read_diagnostic_gate(metadata: dict) -> tuple[dict, dict]:
    report_path = OUTPUT / "diagnostic.json"
    map_path = OUTPUT / "neighbors.npz"
    if not report_path.is_file() or not map_path.is_file():
        raise ValueError("E38 diagnostic and neighbor maps must exist before score stage")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["metadata"] != metadata or sha256(map_path) != report["neighbor_map_sha256"]:
        raise ValueError("E38 diagnostic code, data, preregistration, or map hash changed")
    sparse = report["sparse_recomposed"]
    gate = (sparse["same_prompt_top1_rate_eligible"] >= MIN_SPARSE_TOP1
            and sparse["qualified_neighbor_coverage_eligible"] >= MIN_SPARSE_COVERAGE)
    if not gate or not report["score_stage_allowed"] or not all(report["score_stage_gate"].values()):
        raise ValueError("E38 sparse retrieval gate failed; classification scoring is prohibited")
    with np.load(map_path, allow_pickle=False) as data:
        maps = {name: data[name] for name in data.files}
    return report, maps


def validate_map(indices: np.ndarray, weights: np.ndarray, folds: np.ndarray) -> None:
    n = len(folds)
    if indices.shape != (n, TOP_K) or weights.shape != indices.shape:
        raise ValueError("E38 neighbor map dimensions changed")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("E38 neighbor weights are invalid")
    present = indices >= 0
    if (indices < -1).any() or (indices[present] >= n).any() or np.any(weights[~present] != 0):
        raise ValueError("E38 neighbor indices or empty-slot weights are invalid")
    if (indices[present] == np.repeat(np.arange(n), TOP_K).reshape(n, TOP_K)[present]).any():
        raise ValueError("E38 map includes a self-neighbor")
    if np.any(folds[np.maximum(indices, 0)][present] != np.repeat(folds, TOP_K).reshape(n, TOP_K)[present]):
        raise ValueError("E38 map crosses outer validation folds")
    totals = weights.sum(axis=1)
    if not np.allclose(totals[present.any(axis=1)], 1.0, atol=1e-5, rtol=0):
        raise ValueError("E38 nonempty neighbor weights do not sum to 1")
    if np.any(totals[~present.any(axis=1)] != 0):
        raise ValueError("E38 empty neighbor rows have positive weight")


def centered_scores(scores: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = scores.copy()
    for position in range(TOP_K):
        present = indices[:, position] >= 0
        result[present] -= CENTERING_COEFFICIENT * weights[present, position, None] * scores[indices[present, position]]
    return result


def score(dev: pd.DataFrame, folds: np.ndarray, metadata: dict) -> None:
    diagnostic_report, maps = read_diagnostic_gate(metadata)
    if not np.array_equal(maps["ids"], dev["id"].to_numpy(dtype=str)) or not np.array_equal(maps["folds"], folds):
        raise ValueError("E38 neighbor map does not align with official development rows")
    if set(maps["sparse_panel"].tolist()) != set(SPARSE_PANELS):
        raise ValueError("E38 sparse panel IDs are incomplete")
    for key in ("full", "sparse"):
        validate_map(maps[f"{key}_neighbors"], maps[f"{key}_weights"], folds)
    if (OUTPUT / "score_started.json").exists() or (OUTPUT / "score_metrics.json").exists():
        raise ValueError("E38 score stage was already started")

    # Only after the retrieval gate: open the saved E24 scores and labels.
    from sklearn.metrics import balanced_accuracy_score
    from run_e0 import LABELS
    from run_e37_char4_likelihood import reproduce_e24, source_hashes
    from run_score_normalization import paired_prompt_bootstrap

    source_hashes()
    e24_scores, e24_pred = reproduce_e24(dev, folds)
    labels = np.asarray(LABELS)
    truth = dev["model"].to_numpy()
    if not np.isclose(balanced_accuracy_score(truth, e24_pred), 0.425, atol=1e-12, rtol=0):
        raise ValueError("E24 0.425000 baseline did not reproduce")
    (OUTPUT / "score_started.json").write_text(json.dumps({
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "diagnostic_sha256": sha256(OUTPUT / "diagnostic.json"),
        "e37_reproduction_code_sha256": sha256(ROOT / "run_e37_char4_likelihood.py"),
    }, indent=2) + "\n", encoding="utf-8")

    full_scores = centered_scores(e24_scores, maps["full_neighbors"], maps["full_weights"])
    sparse_scores = centered_scores(e24_scores, maps["sparse_neighbors"], maps["sparse_weights"])
    full_pred = labels[full_scores.argmax(axis=1)]
    sparse_pred = labels[sparse_scores.argmax(axis=1)]
    base_ba = float(balanced_accuracy_score(truth, e24_pred))
    full_ba = float(balanced_accuracy_score(truth, full_pred))
    sparse_ba = float(balanced_accuracy_score(truth, sparse_pred))
    fold_rows = []
    for fold in FOLDS:
        mask = folds == fold
        before = float(balanced_accuracy_score(truth[mask], e24_pred[mask]))
        after_full = float(balanced_accuracy_score(truth[mask], full_pred[mask]))
        after_sparse = float(balanced_accuracy_score(truth[mask], sparse_pred[mask]))
        fold_rows.append({"fold": fold, "e24_ba": before, "full_ba": after_full,
                          "sparse_ba": after_sparse, "full_delta": after_full - before,
                          "sparse_delta": after_sparse - before})
    class_rows = []
    for label in LABELS:
        mask = truth == label
        before = float(np.mean(e24_pred[mask] == label))
        after_full = float(np.mean(full_pred[mask] == label))
        after_sparse = float(np.mean(sparse_pred[mask] == label))
        class_rows.append({"label": label, "e24_recall": before,
                           "full_recall": after_full, "sparse_recall": after_sparse,
                           "full_delta": after_full - before, "sparse_delta": after_sparse - before})
    panel_rows = []
    for panel in SPARSE_PANELS:
        mask = maps["sparse_panel"] == panel
        if mask.sum() != 120 * panel:
            raise ValueError(f"Sparse panel {panel} has unexpected response count")
        supports = {label: int(np.sum(truth[mask] == label)) for label in LABELS}
        all_classes = all(count > 0 for count in supports.values())
        panel_rows.append({"panel_size_per_prompt": panel, "n_rows": int(mask.sum()),
                           "class_support": supports,
                           "e24_ba": float(balanced_accuracy_score(truth[mask], e24_pred[mask]))
                               if all_classes else None,
                           "candidate_ba": float(balanced_accuracy_score(truth[mask], sparse_pred[mask]))
                               if all_classes else None,
                           "ba_defined_for_all_18_classes": all_classes})
    gate = {
        "full_delta_at_least_0_01": full_ba - base_ba >= 0.01 - 1e-12,
        "at_least_4_of_5_full_folds_non_decreasing":
            sum(row["full_delta"] >= -1e-12 for row in fold_rows) >= 4,
        "every_full_class_drop_strictly_less_than_0_05":
            min(row["full_delta"] for row in class_rows) > -0.05 + 1e-12,
        "sparse_ba_not_below_e24": sparse_ba >= base_ba - 1e-12,
    }
    report = {
        "experiment": "e38_soft_prompt_neighbors_score",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "diagnostic_sha256": sha256(OUTPUT / "diagnostic.json"),
        "neighbor_map_sha256": sha256(OUTPUT / "neighbors.npz"),
        "metadata": metadata,
        "e24_source_hashes": source_hashes(),
        "score": {"e24_ba": base_ba, "full_ba": full_ba, "sparse_ba": sparse_ba,
                  "full_delta": full_ba - base_ba, "sparse_delta": sparse_ba - base_ba},
        "folds": fold_rows,
        "classes": class_rows,
        "sparse_panels": panel_rows,
        "paired_prompt_bootstrap_full": paired_prompt_bootstrap(
            truth, dev["prompt_id"].to_numpy(), e24_pred, full_pred, repeats=20000, seed=SEED),
        "paired_prompt_bootstrap_sparse": paired_prompt_bootstrap(
            truth, dev["prompt_id"].to_numpy(), e24_pred, sparse_pred, repeats=20000, seed=SEED),
        "retrieval_gate_passed": diagnostic_report["score_stage_allowed"],
        "score_gate": gate,
        "go": all(gate.values()),
        "holdout_or_test_scored": False,
    }
    predictions = dev.loc[:, ["id", "prompt_id"]].copy()
    predictions["true_model"] = truth
    predictions["fold"] = folds
    predictions["e24_pred"] = e24_pred
    predictions["full_pred"] = full_pred
    predictions["sparse_pred"] = sparse_pred
    with (OUTPUT / "oof_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        predictions.to_csv(stream, index=False)
    with (OUTPUT / "score_metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"score_metrics": str(OUTPUT / "score_metrics.json"),
                      "go": report["go"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "diagnostic", "score"))
    args = parser.parse_args()
    dev, folds, metadata = load_development()
    if args.command == "preflight":
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.command == "score":
        score(dev, folds, metadata)
        return
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError(f"Refusing to overwrite a nonempty diagnostic directory: {OUTPUT}")
    maps, vocabulary_sizes = retrieve_maps(dev, folds)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "neighbors.npz").open("xb") as stream:
        np.savez_compressed(stream, **maps)
    with np.load(OUTPUT / "neighbors.npz", allow_pickle=False) as saved:
        persisted_maps = {name: saved[name] for name in saved.files}
    report = audit_diagnostic(dev, folds, metadata, persisted_maps, vocabulary_sizes,
                              sha256(OUTPUT / "neighbors.npz"))
    with (OUTPUT / "diagnostic.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(OUTPUT / "diagnostic.json"),
                      "n_folds": len(FOLDS), "n_scenarios": len(report["pooled"]),
                      "score_stage_allowed": report["score_stage_allowed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
