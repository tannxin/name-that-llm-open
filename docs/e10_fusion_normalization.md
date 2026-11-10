# E10: fixed score normalization before fusion

The original Kaggle Version 2 submission adds row-wise ranks from E0c and the corrected 12-epoch RoBERTa, with equal weight. This experiment compares that fixed rule with one prespecified, parameter-free alternative: subtract each model's mean across its 18 class scores for a row, divide by that row's standard deviation, and add the two standardized score vectors with equal weight. It does not fit a temperature, weight, or calibrator to development labels.

Both inputs are the saved, aligned out-of-fold score files for the original 120 development prompts (2,160 responses, five GroupKFold folds). `run_score_normalization.py` verifies IDs, labels, fold assignments, score finiteness, and exact reproduction of all 2,160 published rank-fusion predictions before calculating the alternative. It never reads the reserved 30-prompt holdout.

| Fusion rule | Development OOF balanced accuracy | Difference |
| --- | ---: | ---: |
| Equal row ranks (published E7e-R) | 0.383333 | — |
| Equal row-standardized scores | **0.399537** | **+0.016204** |

The standardized rule improves folds 2–5 and is lower on fold 1. Its fold scores are 0.368056, 0.402778, 0.449074, 0.372685, 0.405093, compared with rank scores 0.372685, 0.381944, 0.432870, 0.354167, 0.375000. A paired bootstrap resampling 120 prompts 20,000 times gives a 95% percentile interval of **[+0.001389, +0.031481]** for standardized minus rank balanced accuracy. The same development set has supported several earlier experiments, so this interval does not account for the selection of a promising candidate across the whole project. The locked holdout was not reused.

## Kaggle hidden rerun

The [normalized-score Notebook Version 3](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365857241) completed its hidden rerun and scored **0.401 public balanced accuracy**, compared with **0.376** for the original [rank-fusion Version 2](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365821422). This is a **+0.025 public-score difference** with the same fitted E0c and RoBERTa weights. The 0.399537 and 0.383333 values above are development OOF scores, while the public scores use only the public portion of hidden test data. The private leaderboard result is not yet known.

The largest per-class recall gains are `gpt-5.6-luna` +0.100, `gemini-3.6-flash` +0.075, `claude-opus-4-6` +0.058, and `gpt-5.5` +0.058. `gemini-3.7-flash` falls by 0.100. Full per-class and per-fold files are in the Git-ignored `outputs/e10_fusion_normalization/` directory.

Run from the project root with the saved OOF files:

```bash
python run_score_normalization.py
```

`kaggle/submit_normalized.py` applies the same row-standardization rule to the unchanged full-data model package. The original `final_pipeline.py` and the five source files whose hashes are recorded in that package remain unchanged.
