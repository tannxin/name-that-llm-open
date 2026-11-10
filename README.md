# Name That LLM — reproducible experiments

This repository starts with a local, reproducible baseline for the [Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm/overview/description). The official training file contains `id`, `prompt_id`, `text`, and `model`. The metric is balanced accuracy across 18 exact model names. The competition test prompts differ from the training prompts, so validation groups rows by `prompt_id`.

The [experiment plan](docs/experiment_plan.md) reserves 30 prompts as a locked final holdout and compares models on the other 120 prompts with the same five grouped folds. Development OOF balanced accuracy is 0.2810 for TF-IDF/SVM and 0.3426 after adding fixed style features. A fixed equal-rank blend with a corrected fine-tuned RoBERTa run reaches 0.3833; all five development folds improve over the style-feature baseline. After freezing this choice, its one-time 30-prompt holdout balanced accuracy was **0.4005** on 539 rows. See [results](docs/results.md) for the neural, external-data, ablation, and holdout comparisons. Neither the development nor holdout score is a Kaggle test score.

The original [Kaggle Notebook submission (Version 2)](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365821422) scored **0.376 public balanced accuracy**. The [normalized-score submission (Version 3)](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365857241) completed the hidden rerun and scored **0.401**, a +0.025 public-score difference using the same fitted model weights. The public leaderboard uses about 40% of hidden test rows; the final private score may differ. Submission code is in [`kaggle/`](kaggle/), and the [round-three plan](docs/round3_plan.md) specifies the subsequent grouped experiments.

The next grouped-validation round tested exact punctuation, function-word features, 512-token RoBERTa input, and sibling-version soft experts. Those changes did not show stable gains over the existing pipeline. A parameter-free change to the fusion rule—standardizing each model's 18 scores within each response before adding them equally—reached **0.3995 development OOF**, versus 0.3833 for the original rank fusion. It reuses the same full-data model weights; see the [fusion experiment](docs/e10_fusion_normalization.md) and [offline normalized submission script](kaggle/submit_normalized.py). The 0.3995 OOF and 0.401 public score are separate evaluations.

The [E12 external-data audit](docs/e12_external_data.md) rejected an older open pool for insufficient per-row model provenance. A later [source-level audit](docs/e24_prvsiyan_source_audit.md) accepted a conservative 17-class subset of the prvsiyan v9 data for an exploratory E24 experiment. That source's generation protocol used length statistics from the full official training set, so its grouped OOF is useful for candidate selection but does not establish independent test performance.

The later grouped experiments tested a 512-token frozen Qwen 4B representation, family auxiliary losses, and supervised DeBERTa-v3-base and ModernBERT-base encoders. Three ModernBERT random seeds in a fixed E0c + RoBERTa blend reached **0.4093 development OOF**. The full-data [E16 offline submission](kaggle/README_e16.md) completed the Kaggle hidden rerun with **0.407 public balanced accuracy**; its 4,000-row offline CPU benchmark took 83 minutes 24.582 seconds. The independent family head, within-family margin, sentence-window crop, and several later postprocessing ablations did not meet their advancement criteria. See the [results table](docs/results.md), [three-seed analysis](docs/e16_modernbert_seeds.md), and [submission handoff](docs/e16_kaggle_handoff.md).

The audited external-data E24 candidate reached **0.4250 development OOF**, versus 0.4093 for E16 with the same prompt-grouped folds and fixed fusion. Its [Kaggle Notebook](https://www.kaggle.com/code/tanxin357/name-that-llm-e24-audited-external-e0c-submission?scriptVersionId=367930146) completed the hidden rerun with **0.434 public balanced accuracy**, up 0.027 from E16's 0.407; the private score is unknown. E25 external-data ModernBERT reached 0.4292 when stacked with E24, but the combined gain over E24 was small and inconsistent across folds. Adding 250 audited Qwen 0902 responses in [E29](docs/e29_qwen_sensitivity_results.md) reduced the fixed fusion to 0.4236, so that branch was not promoted. Two further source audits ([E31](docs/e31_independent_source_audit.md), [E32](docs/e32_willrice_raw_audit.md)) admitted no additional training rows under the strict provenance rule. The [current public-method survey](docs/leader_methods_update_2026-11-10.md) and [E30 supervised embedding design](docs/e30_supervised_embedding8b_design.md) guide the next model branch. Development OOF numbers and Kaggle public scores are separate evaluations.

## Get the data

1. Sign in to Kaggle and open the [competition page](https://www.kaggle.com/competitions/name-that-llm/overview).
2. Join the competition and accept its rules, if prompted.
3. Download `train.csv` from the [Data tab](https://www.kaggle.com/competitions/name-that-llm/data).
4. Put it at `data/train.csv` in this project. The `data/` directory is ignored by Git.

No Kaggle credentials or competition data are included in this repository.

## Run the surface baselines

Install Python 3.10 or newer and the four dependencies in `requirements.txt`. Then run:

```bash
python -m pip install -r requirements.txt
python run_e0.py --train data/train.csv --output-dir outputs/e0_tfidf_svm
python run_style.py --train data/train.csv --e0-dir outputs/e0_tfidf_svm --output-dir outputs/e0b_style_only
python run_combined.py --train data/train.csv --e0-dir outputs/e0_tfidf_svm --e0b-dir outputs/e0b_style_only --output-dir outputs/e0c_tfidf_plus_style
```

The CLI checks the schema, nonempty fields, unique `id` values, unique `(prompt_id, model)` pairs, and the exact 18 official labels before training. With the default seed (`42`), it reserves 20% of prompts (30 of the official 150) as a final holdout and writes their IDs to `split_manifest.csv`. It does not calculate holdout predictions or a holdout score. Keep this split fixed while developing models.

On the remaining prompts, it fits word and character TF-IDF features with a balanced linear SVM. Five `GroupKFold` splits keep each prompt entirely in either training or validation. All vectorizers are fit inside each fold, avoiding validation text leakage. The fitted model artifact also uses development rows only.

The output directory contains:

| File | Contents |
| --- | --- |
| `split_manifest.csv` | Fixed prompt IDs assigned to development or the originally reserved holdout |
| `oof_predictions.csv` | One out-of-fold prediction per development row, in input order |
| `fold_metrics.csv` | Balanced accuracy and sizes for each prompt-grouped fold |
| `confusion_matrix.csv` | True-model rows by predicted-model columns |
| `class_metrics.csv` | Support, correct count, and recall for each model |
| `metrics.json` | Development out-of-fold balanced accuracy, fold results, label order, data hash, and package versions |
| `dev_model.joblib` | The same pipeline fitted on development rows after validation |

The overall out-of-fold balanced accuracy is calculated across all development validation predictions. It can differ slightly from the mean of the five fold scores because each class may have a different number of examples in each fold. To reproduce a saved split, use the same `--seed` and `--holdout-fraction` values and output directory; the script rejects a changed split in that directory. Set `--holdout-fraction 0` only if you intentionally want cross-validation over the full training set. Do not inspect holdout performance until the experiment design is final.

This script does not generate a Kaggle submission. The `joblib` file is tied to its Python and scikit-learn environment; retrain in the target environment rather than assuming that the file can be loaded across package versions.

## Neural comparisons

The [model inventory](docs/model_inventory.md) records exact checkpoint IDs, pooling rules, and license caveats. `run_frozen_embeddings.py` and `run_frozen_more.py` evaluate frozen encoders with the saved E0 folds. `run_finetune_encoder.py` fine-tunes MiniLM, RoBERTa, or MPNet with a fixed epoch count and the same folds. Use a CUDA-compatible PyTorch build for neural runs; the server setup and checkpoint provenance are in [E1 notes](docs/e1_frozen_embeddings.md). All scripts keep the 30-prompt holdout out of development training and scoring.

Raw competition files, checkpoints, and experiment outputs are ignored by Git. The [final pipeline](docs/final_pipeline.md) records the locked model choice, one-time holdout check, full-data training, and completed Kaggle submission.

## License

Project code and documentation are released under the [MIT License](LICENSE). Competition data and third-party model weights retain their own licenses and are not included in this repository.
