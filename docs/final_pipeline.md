# Final evaluation and offline submission pipeline

The selected candidate is the fixed equal-rank blend of E0c and the corrected 12-epoch RoBERTa run (E2d-R). Selection was frozen after E4 and before the one-time holdout check. The completed holdout balanced accuracy was **0.4005** on 539 rows from 30 prompts (E0c 0.3483; RoBERTa 0.3802). The choice was not revised from this result. A fresh full-data model package was then trained on all 2,699 official rows from 150 prompts. The commands below reproduce that order.

## Fixed recipe and provenance

- E0c is `run_combined.make_pipeline()`: original-case word 1–2 gram TF-IDF, character 3–5 gram TF-IDF, and 57 standardized then row-normalized style features, with balanced `LinearSVC(C=1)`.
- RoBERTa is `FacebookAI/roberta-base` at commit `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. The official `model.safetensors` SHA-256 is `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`. The final trainer refuses a different base weight file. It uses masked mean pooling, 256 tokens, dropout 0.1, an 18-class linear head, inverse-frequency class weights, batch 16, AdamW encoder/head rates `2e-5`/`1e-3`, weight decay 0.01, 10% linear warmup, seed 42, and exactly 12 epochs. The fp16 scheduler advances only when GradScaler actually updates the optimizer. No validation-based checkpoint selection occurs.
- The base checkpoint must be the `snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b` directory. The trainer checks the exact six-file snapshot and SHA-256 of every file before fitting: `config.json` (`ef0185e2...fc45e9`), `tokenizer.json` (`847bbeab...581df70e`), `tokenizer_config.json` (`994f4675...ecd18ce`), `vocab.json` (`9e7f63c2...e50655`), `merges.txt` (`1ce16647...26adc5`), and `model.safetensors` (full hash above). These hashes were measured from the pinned local Hugging Face snapshot used for the corrected development run. Extra tokenizer files also cause rejection.
- Per row, both models' 18 decision scores are converted to ascending ranks `0..17`. Their ranks are summed, with E0c's rank multiplied by `1e-4` solely to resolve exact sum ties. `run_fixed_blend.py` defines this selected OOF rule; `final_pipeline.py` applies the same rule to unseen rows.
- The corrected development run used Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, PyTorch 2.6.0+cu118, and Transformers 4.51.3. `requirements-final.txt` declares the final pipeline dependencies. For joblib safety, inference requires the same scikit-learn version; PyTorch must have the same base version and Transformers the same exact version as at final fit. Use a CUDA-compatible offline PyTorch wheel for training. `pyarrow` is required to write and verify the submission parquet.

The local Git repository excludes `data/`, `outputs/`, model weights, and parquet files. The final model package must be distributed separately as a Kaggle Dataset or competition-allowed input, together with the exact five source files whose hashes are recorded in package metadata: `final_pipeline.py`, `run_e0.py`, `run_style.py`, `run_combined.py`, and `run_fixed_blend.py`. No network access is used in package inference.

## Run order after selection is frozen

All paths below are relative to the project root. Set `BASE` to the local `models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b` directory; the trainer checks the snapshot name, exact files, and hashes.

```bash
python final_pipeline.py freeze-selection \
  --decision-note "E4 completed; selected E0c plus corrected RoBERTa equal-rank blend on development OOF only." \
  --output outputs/final/selection_lock.json

python final_pipeline.py fit-development \
  --selection-lock outputs/final/selection_lock.json \
  --pretrained-dir "$BASE" \
  --output-dir outputs/final/development_package

python final_pipeline.py evaluate-holdout \
  --selection-lock outputs/final/selection_lock.json \
  --dev-package outputs/final/development_package

python final_pipeline.py fit-full \
  --selection-lock outputs/final/selection_lock.json \
  --holdout-report outputs/final/holdout_report.json \
  --pretrained-dir "$BASE" \
  --output-dir outputs/final/full_package
```

`freeze-selection` checks all three completed E4 conditions (CE, cross-prompt SupCon, and SupCon with same-prompt hard negatives), their OOF metrics and run configs, the E0c and corrected RoBERTa OOF alignment, **exact reproduction of all 2,160 saved E7e-R OOF predictions** by the fixed rank rule, the fixed RoBERTa configuration, and the training file and split hashes. Its output records all input hashes, three E4 scores, and the frozen choice. The development package fits only the 120 development prompts. `evaluate-holdout` is guarded by that package and lock, expects the original 30 prompts/539 rows, and writes one report beside the lock with balanced accuracy, component scores, per-class recall, and the confusion matrix. It refuses an existing report or an alternate report path. A failed inference can be rerun before a report is written; after a report exists, use it as the sole final check. `fit-full` requires the report and then trains a fresh model on all 150 prompts. The full model must never be used to rescore the holdout.

The selection lock and both model packages contain hashes for audit. Training with 2,160 and 2,699 rows respectively changes the steps per epoch; their fitted parameters are not expected to match any one OOF fold checkpoint. The five prompt-grouped folds are for model selection, and the 30-prompt result is a single final estimate.

## Kaggle Notebook, internet disabled

Attach the full package and the exact source files as allowed input to the notebook. The official `test.csv` has `id,text`; the submission has `id,model`. From the directory containing the source files, run:

```bash
python final_pipeline.py submit \
  --full-package /kaggle/input/name-that-llm-final/full_package \
  --test /kaggle/input/competitions/name-that-llm/test.csv \
  --output /kaggle/working/submission.parquet
```

Adjust the model Dataset path to its mounted name. Omit `--sample-submission` during scoring because Kaggle replaces the example test rows with approximately 4,000 hidden rows. The script loads the saved E0c joblib pipeline and trained RoBERTa state dict from local files only. It preserves test row order and checks unique/nonempty IDs, 18 valid labels, and a parquet round trip. The full package exists on the training server at `outputs/final/full_package`. A three-row offline smoke test generated and round-trip-checked `outputs/final/submission_smoke.parquet` using the example sample submission.

## Kaggle submission result

On 2026-11-05, the full package, five exact source files, and a 35-wheel offline dependency set were attached as a private Kaggle Dataset. The [Notebook version 2](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365821422) ran with internet disabled and CPU only, generated a valid three-row `submission.parquet` in the commit run, then succeeded in the competition's full hidden-test rerun. The [Kaggle submissions page](https://www.kaggle.com/competitions/name-that-llm/submissions) reported **0.376 public balanced accuracy**; the public leaderboard showed rank 3,506 of 4,684 teams at the first check. The public score covers approximately 40% of hidden test rows and is not the final private score. It is 0.0073 below development OOF and 0.0245 below the one-time holdout, so further experiments should continue to use the fixed prompt-group validation protocol rather than tune repeatedly to public feedback.

## Completed validation and training

The script was compiled and its CLI help was invoked locally. The rank function exactly reproduced all 2,160 saved E7e-R development OOF predictions. The server runtime was checked for PyTorch 2.6.0+cu118, Transformers 4.51.3, scikit-learn 1.7.2, and support for the offline `AutoModel.from_config(..., add_pooling_layer=False)` constructor. The server now contains `outputs/final/selection_lock.json`, `outputs/final/development_package`, `outputs/final/holdout_report.json`, `outputs/final/full_package`, and the three-row `outputs/final/submission_smoke.parquet`. The development package metadata records 2,160 rows/120 prompts; the full package metadata records 2,699 rows/150 prompts. The holdout report records the single 0.4005108557 balanced-accuracy estimate and component scores above. The smoke test checked test/sample ID order, output schema, and a Parquet round trip with `pyarrow 25.0.1`. It does not measure hidden-test runtime or leaderboard accuracy. Raw data, weights, output packages, and the report remain outside Git.
