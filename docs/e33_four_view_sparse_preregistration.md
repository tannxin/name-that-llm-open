# E33: four-view sparse classifier, locked local CPU probe

Status: pre-registered before the E33 validation run. This experiment tests whether the four complementary sparse views described in the public [V47 notebook](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) improve our official-data E0c and fixed E16 ensemble. This is an independent implementation, not a reproduction of V47's external-data training, function-word vocabulary, or full ensemble.

## Locked data and environment

- Only official `data/train.csv`, SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Preserve the existing E0 30-prompt holdout. Fit and score only the saved 120-prompt/2,160-row development subset in the original five GroupKFold assignments; each fold fits every TF-IDF vocabulary and the SVC on its 96 training prompts only.
- Before fitting E33, rerun the unchanged `run_combined.make_pipeline()` E0c in this same local Python/sklearn environment and compare its per-row predictions and margins to the saved E0c files. If any prediction differs, report all E33 numbers as **exploratory**, even if E33 wins. No Kaggle or production promotion from an environment-mismatched run.
- Never read or score the reserved holdout or test data in this script.

## Fixed features and classifier

The input is the raw answer text without cleaning. A `FeatureUnion` concatenates four L2-normalized TF-IDF views, with weights fixed before validation:

1. Original character 3–5 grams, `max_features=250000`, weight 1.0.
2. Original word/punctuation 1–3 grams, `max_features=180000`, weight 0.7. The token pattern is `(?u)\b\w+\b|[^\w\s]` so punctuation is retained.
3. Shape character 2–6 grams, `max_features=60000`, weight 0.5. Each maximal Unicode-letter span becomes `X` only if the **whole span** is uppercase, otherwise `x`; digits, punctuation and whitespace remain.
4. Function-word/punctuation 1–3 grams, `max_features=50000`, weight 0.5. Tokenize using the same word/punctuation pattern. Keep punctuation and a frozen English function-word list coded in `run_e33_four_view_sparse.py`; replace other word tokens with `CONTENT`, retaining case for function words. This is a small fixed approximation of V47's function-word view, not a claim of byte-exact replication.

All TF-IDF blocks use `min_df=2`, `lowercase=False`, `sublinear_tf=True`, `dtype=float32`. The classifier is `LinearSVC(C=0.5, class_weight='balanced', dual=True, max_iter=5000, random_state=42)`.

## Fixed comparisons and stop gate

Report pooled and five-fold balanced accuracy for E0c rerun, saved E0c, E33 alone, saved E16, and **E16 with its E0c stream replaced by E33**. The E16 reconstruction uses row-z standardized original RoBERTa and three ModernBERT seed score streams; the three ModernBERT streams are averaged as one, following the original E16 rule. The original E16 reconstruction must match all saved E16 predictions. The new fusion adds no fitted weights and uses no labels to choose a formula.

E33 may be considered for a later same-environment server rerun only if (a) local E0c predictions reproduce saved E0c exactly, (b) replacement fusion beats saved E16 by at least +0.010 pooled balanced accuracy, (c) at least 4/5 folds do not decrease, and (d) no class recall drops by 0.05 or more. A failed prerequisite or gate is a no-go for promotion, even if the E33 single-model score rises. Report the worst per-class change and prompt-paired bootstrap interval as diagnostics, not as tuning criteria.

The optional E24 anchor is diagnostic only: replacing E24's sparse E24-external branch with official-only E33 would confound feature design with external-data removal, so this probe does not make that comparison. If E33 merits escalation, a separate pre-registered same-source/fold E24 four-view fit is required.
