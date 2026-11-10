# E36: fixed E24 + E28 + E30 triple fusion (prospective preregistration)

Status at lock: E28 five-fold OOF is complete; its fixed E24+E28 balanced accuracy is 0.434259 versus archived E24 0.425000, a +0.009259 change that narrowly misses the +0.01 gate. E30 has entered its resource stage. **No E30 labeled OOF score or validation result was examined to design this rule.** This is prospective only with respect to E30; the known E28 result is disclosed above.

## Inputs and identity

Use the official E0 development partition only: 2,160 rows, 120 prompts, 18 classes, and its five fixed prompt-grouped folds. Never read the 30-prompt holdout or test labels. Use the archived E24 source scores and saved E24 OOF, with exact file SHA-256 checks from E24 metrics (`301463b680bbf83a46bc106ff273e64db853358e131b29c40410a88f8770a557`). Its saved OOF SHA-256 is `8230d7438eb6c0df7015edc61a54cf880904b85d1f8d599fd549a57e357b4448`. E28 must match its locked five-fold protocol, fold run metadata, fold OOF file hashes, resource report, and completed analysis. E30 must match its v3 locked protocol, resource report, fold run metadata, fold OOF file hashes, and completed analysis. All score files must have identical `id`, `prompt_id`, `true_model`, and `fold` rows. Recompute and match archived E24 predictions exactly and BA 0.425. Recompute and match the saved E28 and E30 fixed-comparator BA. Any mismatch aborts analysis.

The `lock` command saves `docs/e36_fixed_triple_fusion_protocol_lock_v2.json` with code/document hashes and canonical E28/E30 config and protocol hashes before any E30 fold OOF exists. It refuses an existing lock or an existing local E30 OOF file. An initial v1 lock is retained as an audit trail; v2 adds exact SHA-256 anchors for the already completed E28 analysis OOF (`9243bb2002158c3f3078fcbba6f8746db71c75ee2ac2807a6789104b4d310d64`) and metrics (`ae4e073357f7b33946d526d87c42c33fa74e33f551c63c5a97c1dfe21479f98f`), checked read-only on the server before v2. The server still had no E30 OOF at that check. The `evaluate` command requires the unchanged v2 lock, checks all inputs, and is CPU-only. The scripts use normalized text and canonical JSON hashes for code/config/lock portability across Windows and Linux line endings, while raw SHA-256 secures score and result files. The lock records no labels or row-level predictions.

## Fixed score rule

For each row, `row-z` subtracts the mean over 18 class scores and divides by their population standard deviation. Class order is `run_e0.LABELS`; final class uses `argmax`, so first label wins an exact tie. No branch weight, temperature, threshold, class bias, or tie policy is estimated on OOF labels.

Let `B = row-z(E24 E0c) + row-z(RoBERTa) + mean(row-z(ModernBERT seed 42), row-z(seed 17), row-z(seed 73))`. Use the archived E24 E0c score branch, not a new local rerun. Let `Q = row-z(E28 final logits)` and `M = row-z(E30 final logits)`. Evaluate exactly four predictions: E24 `argmax(B)`, E24+E28 `argmax(B+Q)`, E24+E30 `argmax(B+M)`, and triple `argmax(B+Q+M)`. Both new branch weights are exactly 1.0. There is no scan or selection among formulas.

## Fixed report and gate

Report pooled balanced accuracy for all four, each fold's balanced accuracy, each class's recall and support, and triple-versus-each-comparator prediction changes: wrong-to-right, right-to-wrong, and changed wrong-to-different-wrong. For each of the three paired triple comparisons, report a 95% percentile interval from 20,000 prompt-level paired bootstrap resamples with seed 42. The bootstrap is descriptive and does not change the gate.

The strongest fixed comparator is the highest pooled development OOF balanced accuracy among E24, E24+E28, and E24+E30. Exact ties resolve in that stated order. The triple passes only if, against this comparator, pooled BA improves by at least 0.01, at least four of five folds do not decrease, and every class recall drop is strictly less than 0.05. Numerical tolerance follows E28/E30: BA and fold comparisons use `1e-12`; the class rule requires the worst drop to be greater than `-0.05 + 1e-12`. Because comparator selection and evaluation use the same development OOF, a pass is only a candidate for further deployment review, not an unbiased generalization estimate. No holdout or test score enters this decision.

The evaluator writes aggregate `metrics.json`, `fold_metrics.csv`, and `class_metrics.csv` under ignored `outputs/e36_fixed_triple_fusion/`. It does not export row-level predictions or data, and must not run until the pre-E30 lock has been saved.
