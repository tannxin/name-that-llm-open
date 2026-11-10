# E8: exact punctuation and function-word ablations

The existing E0c pipeline is unchanged. `run_fingerprint_ablation.py` compares two independently specified additions on the same 120 development prompts (2,160 answers) and the saved five prompt-grouped folds. The 30-prompt holdout is excluded from fitting, feature extraction, model selection, and scoring. Within each fold, TF-IDF vocabularies and the style scaler are fitted only on the other four folds. All OOF IDs, labels, and fold assignments are checked against the saved E0 files.

The `baseline` variant reran the original E0c on the experiment server with scikit-learn 1.7.2 and reproduced **all 2,160 saved E0c predictions exactly**. Its fixed equal-rank blend with the corrected RoBERTa OOF reproduced all 2,160 previously saved blend predictions; the two CSVs on the server had the same SHA-256 (`4f6d8205f97dc7387c86eff5f9dbb328c2171e41423d15435267fd614bbebca9`). The existing local copy uses different line endings but has identical rows. Local scikit-learn 1.5.2 gave a different fold-1 score, so the reported comparisons use the server environment.

## Prespecified feature additions

- **Exact punctuation** appends ten per-1,000-character rates to the original 57 style features: straight/left-curly/right-curly double quotes; straight/left-curly/right-curly single quotes; em dash, en dash, minus sign; and three ASCII periods. It preserves the original three blocks, one style `StandardScaler`, style row normalization, and balanced linear SVM. The existing combined quote and dash rates remain present.
- **Function words** adds a fourth, row-normalized TF-IDF block of 1–3-grams from a fixed English closed-class list. Open-class words are removed; selected words retain their order. `min_df=2` and `max_features=5,000` are fixed in advance. Each fold reached the 5,000-feature cap; the vocabulary is fitted separately inside that fold. The original E0c feature blocks and classifier remain unchanged.

## Development OOF results

| Variant | E0c balanced accuracy | Difference from original E0c | Fixed blend with corrected RoBERTa | Difference from original blend |
| --- | ---: | ---: | ---: | ---: |
| Original E0c | 0.342593 | — | 0.383333 | — |
| Exact punctuation | 0.343981 | +0.001389 | 0.384722 | +0.001389 |
| Function words | 0.334722 | -0.007870 | 0.387500 | +0.004167 |

The paired 120-prompt bootstrap 95% percentile intervals for the differences were:

| Variant | Single-model difference | Fixed-blend difference |
| --- | ---: | ---: |
| Exact punctuation | [-0.005556, 0.008796] | [-0.006019, 0.009259] |
| Function words | [-0.022685, 0.006944] | [-0.011574, 0.019907] |

The punctuation fold scores were 0.337963, 0.375000, 0.388889, 0.303241, 0.314815. Function-word fold scores were 0.303241, 0.375000, 0.368056, 0.300926, 0.326389. Gains vary by fold, and all difference intervals cross zero. The slight fusion gains are exploratory; selecting a winner after several ablations would make the reported OOF optimistic. Neither variant has enough evidence to replace the current Kaggle submission. The locked holdout is not reused to choose between them.

## Reproduction

Run on the existing experiment server with the original `train.csv` and E0 output directories:

```bash
python3 run_fingerprint_ablation.py --variant baseline --output-dir outputs/e8_fingerprint_baseline
python3 run_fingerprint_ablation.py --variant punctuation --output-dir outputs/e8_fingerprint_punctuation
python3 run_fingerprint_ablation.py --variant function_words --output-dir outputs/e8_fingerprint_function_words
```

Each directory contains OOF predictions, 18 aligned decision scores, fold and class metrics, a confusion matrix, and file hashes. For each candidate, `run_fixed_blend.py` uses its `decision_scores.csv` as the left model and `outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv` as the right model; it applies the original equal-rank rule without fitting weights. `compare_oof.py` computes the prompt-paired intervals, saved in each output directory.

POS n-grams were not run: the server has no offline POS tagger or pinned model. A rule-based stand-in would not be a comparable POS experiment, and adding a new dependency is not warranted by these small, unstable gains.
