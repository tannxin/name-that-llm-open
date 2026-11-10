"""Read-only development OOF length diagnostic; writes aggregate JSON only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "outputs/e0c_tfidf_plus_style/split_manifest.csv"
TRAIN = ROOT / "data/train.csv"
LENGTHS = ROOT / "outputs/round3_sentence_crop_roberta_01/development_lengths.csv"
LENGTH_AUDIT = ROOT / "outputs/truncation_audit/audit.json"
E24_PRED = ROOT / "outputs/e24_prvsiyan_analysis/oof_predictions.csv"
E24_SCORES = ROOT / "outputs/e24_prvsiyan_fit/decision_scores.csv"
ROBERTA = ROOT / "outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv"
MODERN = [
    ROOT / "outputs/e15_modernbert_base_12ep_eager/oof_predictions.csv",
    ROOT / "outputs/e16_modernbert_seed17_eager/oof_predictions.csv",
    ROOT / "outputs/e16_modernbert_seed73_eager/oof_predictions.csv",
]
OUT = ROOT / "docs/e35_length_error_aggregate.json"
KEYS = ["id", "prompt_id", "true_model", "fold"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_z(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=1, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("A score row has zero variance")
    return matrix / scale


def q(values: np.ndarray) -> dict[str, float]:
    return {key: float(np.quantile(values, level)) for key, level in
            [("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95)]}


def length_group_stats(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    part = frame.loc[mask]
    n = len(part)
    class_recall = part.groupby("true_model", sort=True)["correct_e24"].mean()
    return {
        "rows": n,
        "prompts": int(part.prompt_id.nunique()),
        "accuracy": float(part.correct_e24.mean()),
        "macro_recall": float(class_recall.mean()),
        "e16_accuracy": float(part.correct_e16.mean()),
        "e24_minus_e16_accuracy": float(part.correct_e24.mean() - part.correct_e16.mean()),
        "median_fusion_margin": float(part.fusion_margin.median()),
        "wrong": int((~part.correct_e24).sum()),
    }


def bootstrap_by_prompt(frame: pd.DataFrame, repeats: int = 5000) -> dict:
    """One paired prompt resample; no thresholds are selected from OOF labels."""
    prompts = sorted(frame.prompt_id.unique())
    labels = sorted(frame.true_model.unique())
    pmap = {value: i for i, value in enumerate(prompts)}
    cmap = {value: i for i, value in enumerate(labels)}
    counts = np.zeros((len(prompts), len(labels), 2), dtype=np.int16)
    hits = np.zeros_like(counts)
    for row in frame.itertuples(index=False):
        idx = (pmap[row.prompt_id], cmap[row.true_model], int(row.truncated_256))
        counts[idx] += 1
        hits[idx] += int(row.correct_e24)
    weights = np.random.default_rng(42).multinomial(
        len(prompts), np.full(len(prompts), 1 / len(prompts)), size=repeats
    )
    boot_count = (weights @ counts.reshape(len(prompts), -1)).reshape(repeats, len(labels), 2)
    boot_hit = (weights @ hits.reshape(len(prompts), -1)).reshape(repeats, len(labels), 2)
    pooled = boot_hit.sum(axis=1) / boot_count.sum(axis=1)
    pooled_diff = pooled[:, 1] - pooled[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = boot_hit / boot_count
    macro_diff = np.nanmean(recall[:, :, 1], axis=1) - np.nanmean(recall[:, :, 0], axis=1)
    return {
        "unit": "official_development_prompt_id",
        "repeats": repeats,
        "seed": 42,
        "long_minus_short_accuracy_95pct": [float(x) for x in np.quantile(pooled_diff, [.025, .975])],
        "long_minus_short_macro_recall_95pct": [float(x) for x in np.quantile(macro_diff, [.025, .975])],
    }


def main() -> None:
    with SPLIT.open(encoding="utf-8", newline="") as handle:
        split = list(csv.DictReader(handle))
    development = {row["prompt_id"] for row in split if row["split"] == "development"}
    assert len(split) == 150 and len(development) == 120
    assert sum(row["split"] == "holdout" for row in split) == 30

    # Parse the CSV stream, but discard nondevelopment rows before accessing
    # their answer text or labels. Nothing about holdout rows is retained.
    dev_rows = []
    with TRAIN.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["prompt_id"] not in development:
                continue
            dev_rows.append({
                "id": row["id"], "prompt_id": row["prompt_id"],
                "true_model": row["model"], "word_count": len(row["text"].split()),
                "char_count": len(row["text"]),
            })
    dev = pd.DataFrame(dev_rows)
    assert len(dev) == 2160 and dev.id.is_unique and dev.prompt_id.nunique() == 120
    assert dev.true_model.value_counts().eq(120).all()

    lengths = pd.read_csv(LENGTHS, dtype={"id": str})
    assert len(lengths) == 2160 and lengths.id.is_unique
    dev = dev.merge(lengths, on="id", validate="one_to_one", suffixes=("", "_cached"))
    assert dev.word_count.eq(dev.word_count_cached).all()
    assert int((dev.token_count > 256).sum()) == 256
    audit = json.loads(LENGTH_AUDIT.read_text(encoding="utf-8"))
    assert audit["source"]["split_manifest_sha256"] == sha256(SPLIT)
    prior_by_class = {row["model"]: row["truncated_256"] for row in audit["by_class"]}
    actual_by_class = dev.assign(truncated=dev.token_count.gt(256)).groupby("true_model").truncated.sum()
    assert actual_by_class.to_dict() == prior_by_class

    predictions = pd.read_csv(E24_PRED, dtype=str, keep_default_na=False)
    assert len(predictions) == 2160 and predictions.id.is_unique
    frame = predictions.merge(dev.drop(columns="word_count_cached"), on="id", validate="one_to_one")
    assert frame.prompt_id_x.eq(frame.prompt_id_y).all()
    assert frame.true_model_x.eq(frame.true_model_y).all()
    frame = frame.rename(columns={"prompt_id_x": "prompt_id", "true_model_x": "true_model"})
    frame = frame.drop(columns=["prompt_id_y", "true_model_y"])
    assert set(frame.prompt_id) == development
    prior_by_fold = {str(row["fold"]): row["truncated_256"] for row in audit["by_fold"]}
    actual_by_fold = frame.assign(truncated=frame.token_count.gt(256)).groupby("fold").truncated.sum()
    assert actual_by_fold.to_dict() == prior_by_fold
    frame["correct_e24"] = frame.pred_e16_with_e24.eq(frame.true_model)
    frame["correct_e16"] = frame.pred_e16.eq(frame.true_model)
    frame["truncated_256"] = frame.token_count.gt(256)

    score_paths = [E24_SCORES, ROBERTA, *MODERN]
    score_frames = [pd.read_csv(path, dtype={key: str for key in KEYS}) for path in score_paths]
    score_cols = [col for col in score_frames[0] if col.startswith("score_")]
    assert len(score_cols) == 18
    labels = np.array([col[6:] for col in score_cols])
    matrices = []
    for score_frame in score_frames:
        assert score_frame.id.is_unique and set(score_frame.id) == set(frame.id)
        score_frame = frame[KEYS].merge(score_frame, on="id", validate="one_to_one", suffixes=("", "_s"))
        for key in KEYS[1:]:
            assert score_frame[key].astype(str).eq(score_frame[key + "_s"].astype(str)).all()
        assert all(col in score_frame for col in score_cols)
        matrices.append(row_z(score_frame[score_cols].to_numpy(dtype=np.float64)))
    fusion = matrices[0] + matrices[1] + np.mean(np.stack(matrices[2:]), axis=0)
    pred = labels[fusion.argmax(axis=1)]
    assert np.array_equal(pred, frame.pred_e16_with_e24.to_numpy())
    frame["fusion_margin"] = np.sort(fusion, axis=1)[:, -1] - np.sort(fusion, axis=1)[:, -2]

    short = ~frame.truncated_256.to_numpy()
    long = frame.truncated_256.to_numpy()
    short_stats, long_stats = length_group_stats(frame, short), length_group_stats(frame, long)
    class_rows = []
    for label in sorted(frame.true_model.unique()):
        part = frame[frame.true_model == label]
        a = part[~part.truncated_256]
        b = part[part.truncated_256]
        class_rows.append({
            "class": label, "short_rows": len(a), "long_rows": len(b),
            "short_recall": float(a.correct_e24.mean()), "long_recall": float(b.correct_e24.mean()),
            "long_minus_short_recall": float(b.correct_e24.mean() - a.correct_e24.mean()),
        })
    prompt_rows = []
    for _, part in frame.groupby("prompt_id"):
        a, b = part[~part.truncated_256], part[part.truncated_256]
        prompt_rows.append({"n_long": len(b), "acc": float(part.correct_e24.mean()),
                            "within_delta": float(b.correct_e24.mean() - a.correct_e24.mean())
                            if len(a) and len(b) else None})
    mixed = [row["within_delta"] for row in prompt_rows if row["within_delta"] is not None]
    high_margin = frame.fusion_margin.ge(frame.fusion_margin.quantile(.75))

    # Quartiles are determined by answer lengths alone, before looking at errors.
    quartiles = {}
    for length_name in ["char_count", "word_count"]:
        groups = pd.qcut(frame[length_name], q=4, duplicates="drop")
        quartiles[length_name] = []
        for interval, part in frame.groupby(groups, observed=True):
            quartiles[length_name].append({"range": str(interval), "rows": len(part),
                                           "accuracy": float(part.correct_e24.mean()),
                                           "macro_recall": float(part.groupby("true_model").correct_e24.mean().mean())})
    fold_cut = []
    for fold, part in frame.groupby("fold"):
        a, b = part[~part.truncated_256], part[part.truncated_256]
        fold_cut.append({"fold": int(fold), "short_rows": len(a), "long_rows": len(b),
                         "short_accuracy": float(a.correct_e24.mean()),
                         "long_accuracy": float(b.correct_e24.mean()),
                         "long_minus_short_accuracy": float(b.correct_e24.mean() - a.correct_e24.mean())})

    result = {
        "scope": "E24 saved out-of-fold predictions, 120 official development prompts only",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in [SPLIT, LENGTHS, LENGTH_AUDIT, E24_PRED, *score_paths]},
        "train_sha256_from_prior_length_audit": audit["source"]["train_sha256"],
        "roberta_tokenizer_sha256_from_prior_audit": audit["source"]["tokenizer_json_sha256"],
        "n_rows": len(frame), "n_prompts": frame.prompt_id.nunique(), "n_classes": frame.true_model.nunique(),
        "e24_correct": int(frame.correct_e24.sum()), "e16_correct": int(frame.correct_e16.sum()),
        "overall_accuracy_and_ba": float(frame.correct_e24.mean()),
        "length_quantiles": {name: q(frame[name].to_numpy()) for name in
                             ["char_count", "word_count", "token_count"]},
        "token_cut": {"definition": "RoBERTa exact count including special tokens > 256",
                      "short": short_stats, "long": long_stats,
                      "long_minus_short_accuracy": float(long_stats["accuracy"] - short_stats["accuracy"]),
                      "long_minus_short_macro_recall": float(long_stats["macro_recall"] - short_stats["macro_recall"])},
        "prompt_bootstrap": bootstrap_by_prompt(frame),
        "length_quartiles": quartiles,
        "fold_token_cut": fold_cut,
        "by_class_token_cut": class_rows,
        "prompt_context": {"mixed_length_prompts": len(mixed),
                           "long_count_per_prompt_min": min(row["n_long"] for row in prompt_rows),
                           "long_count_per_prompt_max": max(row["n_long"] for row in prompt_rows),
                           "mean_within_prompt_long_minus_short_accuracy": float(np.mean(mixed)),
                           "prompts_long_better": sum(value > 0 for value in mixed),
                           "prompts_long_worse": sum(value < 0 for value in mixed),
                           "prompts_equal": sum(value == 0 for value in mixed),
                           "spearman_long_share_vs_prompt_accuracy": float(pd.DataFrame(prompt_rows).n_long.corr(
                               pd.DataFrame(prompt_rows).acc, method="spearman"))},
        "fusion_margin": {
            "interpretation": "top-two margin of the fixed sum of row-standardized class scores; not calibrated probability",
            "all": q(frame.fusion_margin.to_numpy()),
            "short_correct_median": float(frame.loc[short & frame.correct_e24, "fusion_margin"].median()),
            "short_wrong_median": float(frame.loc[short & ~frame.correct_e24, "fusion_margin"].median()),
            "long_correct_median": float(frame.loc[long & frame.correct_e24, "fusion_margin"].median()),
            "long_wrong_median": float(frame.loc[long & ~frame.correct_e24, "fusion_margin"].median()),
            "top_quartile_margin_threshold_unlabeled": float(frame.fusion_margin.quantile(.75)),
            "high_margin_wrong_short": int((high_margin & short & ~frame.correct_e24).sum()),
            "high_margin_wrong_long": int((high_margin & long & ~frame.correct_e24).sum()),
            "high_margin_rows_short": int((high_margin & short).sum()),
            "high_margin_rows_long": int((high_margin & long).sum()),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["n_rows", "e24_correct", "token_cut",
                                                  "prompt_bootstrap", "length_quartiles", "fold_token_cut",
                                                  "prompt_context", "fusion_margin"]}, indent=2))


if __name__ == "__main__":
    main()
