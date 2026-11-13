# E42 Qwen3-14B-Base: Recovery2 fivefold audit

## Result

The locked fivefold development experiment is complete and independently
audited. Its fixed E24 + 1×E42 fusion reached **0.443519 balanced accuracy**,
versus **0.425000** for E24 alone, a **+0.018519** gain. All five folds were
nondecreasing, and the largest class recall decline was **−0.033333**
(`claude-sonnet-5`). The 20,000-resample paired prompt bootstrap 95% interval
for the gain was **[+0.009259, +0.027778]**. E42 passes the preregistered
E24-relative offline gate.

E42 alone scored 0.390741. The comparisons with earlier fixed blends are
context, not a basis for tuning: E24+E42 was 0.009259 above E24+E28 but
failed the conservative class gate; 0.005093 below E24+E30-A; and only
0.000463 above E24+E39 with three of five folds nondecreasing. No blend
weight or protocol was changed after seeing these results.

## Integrity and resources

The audit verified all 2,160 unique out-of-fold rows (five folds of 432),
120 development prompts, 18 classes, and isolation of the 30 already used
holdout prompts. It independently recomputed the fixed fusion, fold and class
recalls, error transitions, and prompt bootstrap from the row-level snapshot.
It checked the frozen data, model, training code, protocol and resource
hashes, as well as the byte-for-byte reuse of the first three folds.

All five fold resource records passed the locked limits: 643 successful
optimizer updates per fold, about 5,005–5,079 seconds of training per fold,
at most 11.78 GiB reserved during training and 17.20 GiB during validation
inference. Total fold training time was 25,224.79 seconds. GPU1 was released
after completion.

The original run was interrupted by node ephemeral-storage eviction during
fold 1. Recovery1 completed folds 1–3, then stopped in fold 4 when a shared
filesystem status write returned `EAGAIN`. Recovery2 reused the three
completed folds unchanged, trained folds 4–5 with the same locked protocol,
and completed one aggregate analysis. All interruption and recovery evidence
was retained and checked before the audit opened score rows.

The aggregate audit JSON is
`docs/e42_qwen3_14b_base_recovery2_fivefold_audit_aggregate.json`. Training
data, weights, and row-level OOF are excluded from Git.

## Decision

**Offline E24-relative gate: pass. Deployment gate: pending.** A single 16 GB
Kaggle T4 cannot be assumed sufficient for the observed 17.20 GiB inference
peak. Kaggle T4×2, Internet-off, full input loading and the nine-hour limit
require a separate unscored deployment test before any full-data adapter
training or competition submission. The public leaderboard best remains the
previously submitted E24 score of 0.434.
