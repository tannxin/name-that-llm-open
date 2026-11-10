# E33 four-view sparse CPU probe: result

**Decision: no-go for promotion.** The local four-view classifier improved its standalone OOF score over the E0c sparse baseline, but replacing E16's E0c stream reduced the fixed ensemble score. The local Python/sklearn environment also failed to reproduce six saved E0c predictions, so all new scores are exploratory. No holdout, hidden test, Kaggle submission, or E24 external-data branch was evaluated.

The [pre-registration](e33_four_view_sparse_preregistration.md) was written before E33 fitting and has SHA-256 `14340b8f24c4d552b9d11a0bcb584ed6b934abb47de442ca500893566ae78f56`, recorded in the [metrics artifact](e33_artifacts/metrics.json). Implementation: [`run_e33_four_view_sparse.py`](../run_e33_four_view_sparse.py). The run used official `train.csv` SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`, the original 120 prompt/2,160 row development partition, five 24-prompt validation folds, Python 3.12.4 and scikit-learn 1.5.2. Original saved E0c used Python 3.10.12 and scikit-learn 1.7.2.

## Pooled OOF balanced accuracy

| Candidate | OOF BA |
| --- | ---: |
| Saved E0c | 0.342593 |
| Local same-environment E0c rerun | 0.341667 |
| E33 four-view alone | **0.360185** |
| Saved E16, independently reconstructed and exact-prediction checked | **0.409259** |
| E16 with E0c sparse stream replaced by E33 | 0.403241 |

E33's single-model improvement over the local E0c is `+0.018519`; its fixed fusion change from E16 is **`-0.006019`**. The prompt-paired bootstrap 95% interval for the fusion change is `[-0.019444, +0.007407]` (20,000 prompt resamples, seed 42). E0c rerun changed six row predictions relative to saved E0c, with maximum absolute score difference `0.017869`; the original E16 score reconstruction matched every saved E16 prediction.

## Five folds

Each fold has 432 validation rows, 24 validation prompts and all 18 classes.

| Fold | Local E0c | E33 alone | Saved E16 | E16 with E33 replacing sparse |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.324074 | 0.340278 | 0.391204 | 0.386574 |
| 2 | 0.372685 | 0.386574 | 0.425926 | 0.409722 |
| 3 | 0.395833 | 0.407407 | 0.435185 | 0.414352 |
| 4 | 0.305556 | 0.344907 | 0.375000 | 0.386574 |
| 5 | 0.310185 | 0.321759 | 0.418981 | 0.418981 |

Only 2/5 fusion folds are nondecreasing relative to E16. Exact unrounded fold values are in the [fold artifact](e33_artifacts/fold_metrics.csv).

## Class recall

Each class has 120 development responses. Values below are rounded to six decimals; exact values are in the [class artifact](e33_artifacts/class_metrics.csv).

| Class | E33 alone | E16 | E16 with E33 replacing sparse |
| --- | ---: | ---: | ---: |
| gpt-5.5 | 0.333333 | 0.333333 | 0.316667 |
| gpt-5.6-luna | 0.275000 | 0.466667 | 0.450000 |
| gpt-5.6-sol | 0.191667 | 0.191667 | 0.183333 |
| gpt-5.6-terra | 0.241667 | 0.300000 | 0.300000 |
| gpt-6-astra | 0.658333 | 0.741667 | 0.708333 |
| gemini-3.1-pro-preview | 0.808333 | 0.800000 | 0.808333 |
| gemini-3.6-flash | 0.375000 | 0.566667 | 0.550000 |
| gemini-3.7-flash | 0.291667 | 0.325000 | 0.308333 |
| gemini-3.8-flash | 0.416667 | 0.300000 | 0.291667 |
| claude-fable-5 | 0.133333 | 0.175000 | 0.183333 |
| claude-opus-4-6 | 0.491667 | 0.583333 | 0.550000 |
| claude-opus-4-8 | 0.200000 | 0.375000 | 0.375000 |
| claude-opus-5 | 0.616667 | 0.583333 | 0.600000 |
| claude-sonnet-5 | 0.383333 | 0.350000 | 0.350000 |
| qwen3.8-max | 0.483333 | 0.500000 | 0.558333 |
| glm-5.3 | 0.175000 | 0.208333 | 0.233333 |
| glm-5.3-flash | 0.075000 | 0.291667 | 0.216667 |
| kimi-k3 | 0.333333 | 0.275000 | 0.275000 |

The worst fusion class change is `glm-5.3-flash`, from `0.291667` to `0.216667`, a drop of **0.075**. All four pre-registered gate checks fail: baseline reproduction; +0.01 fusion gain; at least four nondecreasing folds; and no class recall drop of 0.05 or more.

This tests an official-data four-view approximation only. It does not test the public V47 external corpus, its exact function-word vocabulary, or the full public ensemble. An E24-specific comparison would need the same audited external rows, same weighting and same folds for both sparse models, separately pre-registered; substituting the official-only E33 scores for E24's external-data sparse stream would confound the feature and data changes.
