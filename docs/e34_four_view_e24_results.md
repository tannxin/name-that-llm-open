# E34 same-source E24 four-view sparse probe: result

**Decision: no-go for promotion.** With the same audited E24 external rows, weights, and folds, the four-view sparse branch raised standalone OOF balanced accuracy by `+0.006019` against a local rerun of E24, but reduced the fixed fusion by `-0.008333`. Only one of five fusion folds did not decline, and two classes lost `0.075` recall. The local E24 rerun also differed from the archived server predictions, so all new numbers are exploratory.

The [pre-registration](e34_four_view_e24_preregistration.md) was written before any E34 fit or validation score. The CPU-only `prepare` stage locked its SHA-256 `e5918b962746e1567f003d3b7a8bf55c53cfb46d019ebd4047706bb6cc976c54`, the [implementation](../run_e34_four_view_e24.py), E24 and E33 source scripts, official/external data, audit files, selected-row manifest, folds, and all fixed-fusion score files. The [protocol lock](e34_artifacts/protocol_lock.json) has SHA-256 `4a31b2fe3206b0b3b958dbd3397c61a03ff67866c6191fd2c8dae7f9c00e49dc`. The archived E24 fixed fusion was independently reconstructed row for row at BA `0.425` before either new fit. E34 then completed the entire local E24 baseline before fitting any four-view model.

## Same-source, same-environment comparison

Both sparse branches used the same 1,728 official development training rows plus the same 3,620 audited external rows in each fold. External weights are exactly E24's selected sample weights, totaling 24 per covered class. Both use the same five 24-prompt/432-row validation folds. The E34 model imports the exact corrected `run_e33_four_view_sparse.make_four_view_pipeline()`; no E33 files were changed. The fixed fusion changes only the sparse scores; archived RoBERTa and three ModernBERT seed streams remain fixed.

| Candidate | Pooled OOF balanced accuracy |
| --- | ---: |
| Archived E24 sparse | 0.372685 |
| Local E24 sparse rerun | 0.369907 |
| E34 four-view sparse | **0.375926** |
| Archived E24 fixed fusion | 0.425000 |
| Local E24 fixed fusion | **0.426389** |
| E34 four-view fixed fusion | 0.418056 |

The local E24 sparse rerun changed **117 of 2,160** archived predictions; the largest absolute margin difference was `0.135864`. The original server E24 run used Python 3.10.12/scikit-learn 1.7.2; this local run used Python 3.12.4/scikit-learn 1.5.2. The discrepancy establishes an environment mismatch; its exact cause was not isolated. The same-environment feature effect is therefore the E34 comparison against **local** E24, and it remains exploratory. The original Prvsiyan source protocol's use of global official length statistics is an additional inherited E24 caveat.

## Five folds

| Fold | Local E24 sparse | E34 sparse | Local E24 fusion | E34 fusion | Fusion change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.351852 | 0.351852 | 0.400463 | 0.384259 | -0.016204 |
| 2 | 0.370370 | 0.402778 | 0.421296 | 0.418981 | -0.002315 |
| 3 | 0.407407 | 0.393519 | 0.465278 | 0.435185 | -0.030093 |
| 4 | 0.344907 | 0.358796 | 0.405093 | 0.416667 | +0.011574 |
| 5 | 0.375000 | 0.372685 | 0.439815 | 0.435185 | -0.004630 |

The prompt-paired 20,000-repeat bootstrap interval for the fixed-fusion difference is **[-0.021759, +0.005093]** (seed 42). Replacing the sparse branch corrected 79 previously wrong fusion predictions and made 97 previously correct predictions wrong; 324 predictions changed in total. These diagnostics did not alter the fixed gate.

## Class recall and pre-registered gate

Each class has 120 official development responses. The largest declines are `gpt-5.6-terra` and `claude-opus-4-6`, both `-0.075`.

| Class | Local E24 fusion | E34 fusion | Change |
| --- | ---: | ---: | ---: |
| gpt-5.5 | 0.366667 | 0.350000 | -0.016667 |
| gpt-5.6-luna | 0.500000 | 0.491667 | -0.008333 |
| gpt-5.6-sol | 0.225000 | 0.233333 | +0.008333 |
| gpt-5.6-terra | 0.333333 | 0.258333 | -0.075000 |
| gpt-6-astra | 0.758333 | 0.750000 | -0.008333 |
| gemini-3.1-pro-preview | 0.816667 | 0.825000 | +0.008333 |
| gemini-3.6-flash | 0.591667 | 0.583333 | -0.008333 |
| gemini-3.7-flash | 0.325000 | 0.291667 | -0.033333 |
| gemini-3.8-flash | 0.316667 | 0.341667 | +0.025000 |
| claude-fable-5 | 0.208333 | 0.233333 | +0.025000 |
| claude-opus-4-6 | 0.666667 | 0.591667 | -0.075000 |
| claude-opus-4-8 | 0.391667 | 0.366667 | -0.025000 |
| claude-opus-5 | 0.600000 | 0.591667 | -0.008333 |
| claude-sonnet-5 | 0.375000 | 0.375000 | +0.000000 |
| qwen3.8-max | 0.500000 | 0.566667 | +0.066667 |
| glm-5.3 | 0.191667 | 0.225000 | +0.033333 |
| glm-5.3-flash | 0.250000 | 0.225000 | -0.025000 |
| kimi-k3 | 0.258333 | 0.225000 | -0.033333 |

All four pre-registered conditions fail: exact local E24 reproduction, `+0.010` fixed-fusion gain, at least four nondecreasing folds, and no class recall drop of `0.05` or more. The [baseline metrics](e34_artifacts/baseline_metrics.json) and [candidate metrics](e34_artifacts/metrics.json) preserve exact values. Full row-level local OOF files are retained under Git-ignored `outputs/e34_four_view_e24/`. No holdout, hidden test, GPU, SSH server, or Kaggle submission was used.
