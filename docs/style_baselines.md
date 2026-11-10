# Style-feature controls (E0b and E0c)

Both experiments use the official `train.csv` and require the saved E0 `split_manifest.csv`, `oof_predictions.csv`, and `metrics.json`. The scripts verify the training file hash, align each development row with the E0 prediction by ID, and reuse E0's exact five prompt-grouped folds. The 30 reserved prompt groups are excluded before feature extraction and are not scored.

## E0b: style-only control

`run_style.py` extracts 57 fixed features from each response. They cover response and word length; line, paragraph, and sentence structure; Unicode letter, digit, punctuation, and symbol fractions; punctuation rates; and formatting such as Markdown headings, lists, blockquotes, code fences, indentation, and blank lines. It also uses a word-type ratio, but never stores or learns a word identity. A `StandardScaler` is fitted on each fold's training rows, followed by a balanced linear SVM with `C=1`. The saved development model is fitted only on the 120 development prompts.

```bash
python run_style.py --train data/train.csv --e0-dir outputs/e0_tfidf_svm --output-dir outputs/e0b_style_only
```

On the verified official file (SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`), E0b reached **0.262037** out-of-fold balanced accuracy, compared with E0's **0.281019** on the same 2,160 development rows. This is a diagnostic baseline; it is not the preferred candidate by itself.

## E0c: TF-IDF plus style

`run_combined.py` adds the E0b features to E0's word and character TF-IDF blocks. Within each fold, it fits the TF-IDF vocabularies and a style `StandardScaler` using training rows only. It then L2-normalizes the style block per response, giving it approximately the same scale as each of the two TF-IDF blocks. The same balanced linear SVM and `C=1` are used. There is no validation-driven weight search.

```bash
python run_combined.py --train data/train.csv --e0-dir outputs/e0_tfidf_svm --e0b-dir outputs/e0b_style_only --output-dir outputs/e0c_tfidf_plus_style
```

E0c reached **0.342593** out-of-fold balanced accuracy, a **+0.061574** absolute difference from E0. All five folds improved. Its strongest class-recall gains versus E0 were Claude Opus 4.6 (+0.258), GPT 6 Astra (+0.192), and Qwen 3.8 Max (+0.192). Claude Fable 5 remained weak (0.033 recall), so this does not resolve all class confusions.

Each output directory contains row-level OOF predictions, fold metrics, per-class recall, a confusion matrix, metrics metadata, a copy of the split manifest, and a development-only fitted model. These results are development estimates; the reserved holdout has not been evaluated.
