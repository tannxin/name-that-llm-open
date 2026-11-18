# Qwen3-8B with prvsiyan v9: five-fold result

**Decision: No-go.** The fixed E24 + v9-mixed Qwen fusion reached balanced accuracy (BA) **0.447222**, compared with **0.443056** for E24 + the official-only E39 Qwen. The gain of **0.004167** is below the preregistered minimum of **0.01**. Four of five folds did not decline, but `qwen3.8-max` recall fell by **0.083333**, beyond the allowed decline of strictly less than **0.05**. The 20,000-replicate paired prompt bootstrap 95% interval for the BA gain was **[−0.005556, +0.013889]**.

The independent audit verified 2,160 unique out-of-fold rows, five folds of 432 rows, 120 development prompts, exclusion of the 30 previously used holdout prompts, the locked source/model/protocol hashes, per-fold resources, and all 18 class recalls. The exploratory mixed-model single-route BA was 0.414352; the official-only E39 single-route BA was 0.356481. This single-route difference did not translate into enough gain for the fixed fusion.

The external v9 source was constructed using global length/style statistics from official data, so this result is **exploratory**, even though the 30 holdout prompts were excluded from model evaluation. We will not fit a full-data adapter or submit this candidate to Kaggle. The best submitted public score remains E24 + E39 at **0.445**; that leaderboard value is a separate measurement and was not used to choose this experiment.

The first audit script rejected valid key rows because its pandas comparison retained different indexes after alignment. The reviewed V2 auditor resets the expected key index before that comparison. It does not change the data, fusion, metrics, or decision thresholds. Its static and mutation tests passed before the five-fold audit was rerun. The aggregate report contains no row-level predictions, answers, model weights, or server paths.

See [`q8_v9_mixed_v1_fivefold_audit_aggregate.json`](q8_v9_mixed_v1_fivefold_audit_aggregate.json) for per-fold and per-class aggregates and the input hashes.
