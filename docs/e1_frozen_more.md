# E1 additional frozen encoders

Run `python run_frozen_more.py` after E0 has produced `outputs/e0_tfidf_svm/`. The script reads E0's saved prompt split and fold assignments, rejects a changed training CSV, and never embeds or scores the 30 reserved holdout prompts. It keeps the original text and case, defaults to a 256-token cap, uses attention-masked mean pooling followed by L2 normalization, and fits the same balanced linear SVM head in each of the five prompt-grouped folds. A 512-token ablation requires `--max-length 512` and a separate `--output-dir`.

| Slug | Hugging Face repository | Pinned commit | Role | Repository license metadata (2026-11-04) |
| --- | --- | --- | --- | --- |
| `all-mpnet-base-v2` | [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | Sentence embedding, recommended mean pooling | Apache-2.0 |
| `roberta-base-raw-mlm` | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | Raw MLM diagnostic with predefined masked mean pooling | MIT |
| `style-embedding` | [`AnnaWegmann/Style-Embedding`](https://huggingface.co/AnnaWegmann/Style-Embedding) | `d7d0f5ca829316a8f5695e49dfce80b86db5e76c` | Style representation, model-card mean pooling | No license field in current API metadata; evaluate only |
| `mpnet-base-raw` | [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) | `6996ce1e91bd2a9c7d7f61daec37463394f73f09` | Raw pretrained encoder diagnostic with predefined masked mean pooling | No license field in current API metadata; evaluate only |

The two sentence embedding cards specify masked mean pooling. The raw RoBERTa and microsoft/mpnet-base encoders have no trained sentence pooling head in this comparison, so their frozen pooled scores are diagnostics rather than like-for-like sentence embedding comparisons. The first-round cap of 256 is below the model or config limits (384 for all-mpnet; 512 for the other three). Per-model results record the number of truncated rows, longest pre-truncation token count, source endpoint, exact commit, and environment versions.

Large public weights may be fetched through a mirror when direct Hugging Face transfer stalls. Pin the official commit and verify the downloaded weight SHA-256 against the official repository file page before using the weight. No model weights, training CSV, or embedding cache belong in Git.

## Development results

The values below use all 2,160 development rows and the five saved E0 folds. The reserved 30 prompt groups are excluded throughout. E0 TF-IDF + linear SVM scored 0.281019 on the same folds.

| Model | Pooled OOF balanced accuracy | Rows truncated at 256 tokens | Weight SHA-256 |
| --- | ---: | ---: | --- |
| `all-mpnet-base-v2` | 0.135185 | 265 | `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658` |
| `roberta-base-raw-mlm` | 0.257870 | 256 | `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81` |
| `style-embedding` | 0.232870 | 256 | `3186cd80660a7169a911bace4d54416cf5771a319a22f84c3a79a961ecb0c6f5` |
| `mpnet-base-raw` | 0.282870 | 265 | `f8eb85dab52341af018fb8cd325a9d83fdb00871c1b2cb1d445d42b289c1d45e` |

The all-mpnet, RoBERTa, and raw MPNet safetensors files came from ModelScope mirror URLs and matched the SHA-256 values on the official Hugging Face repository file pages. Style-Embedding's PyTorch weight came directly from its pinned official Hugging Face commit and matched the official LFS SHA-256. The raw RoBERTa and MPNet model loaders warn that their pooler weights are newly initialized. These evaluations use the encoders' `last_hidden_state` and attention-masked mean; they do not use pooler outputs. Raw MPNet's 0.001852 advantage over E0 is too small to establish a reliable improvement without the planned paired prompt bootstrap. See each model's `metrics.json` and OOF files under `outputs/e1_frozen_more/` for the full reproducibility record.

## Fixed 512-token ablation of raw encoders

The two raw encoders were rerun with `--max-length 512 --batch-size 8 --output-dir outputs/e1_frozen_more_512`. All other settings, model revisions, weights, normalization, classifier, and the 2,160 development rows in E0's five folds stayed fixed. The 30 reserved prompt groups were excluded. Each `metrics.json` records its exact source URL, model commit, weight SHA-256, token statistics, and fold scores. These are frozen encoder diagnostics, not supervised fine-tuning results.

| Model | 256-token OOF | 512-token OOF | Change | Truncated at 256 / 512 | Longest input before truncation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `roberta-base-raw-mlm` | 0.257870 | 0.262500 | +0.004630 | 256 / 0 | 387 tokens |
| `mpnet-base-raw` | 0.282870 | 0.281481 | -0.001389 | 265 / 0 | 381 tokens |

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `roberta-base-raw-mlm` at 512 | 0.259259 | 0.291667 | 0.245370 | 0.254630 | 0.261574 |
| `mpnet-base-raw` at 512 | 0.296296 | 0.296296 | 0.268519 | 0.291667 | 0.254630 |

RoBERTa used pinned commit `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, SHA-256 `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`, and the [ModelScope RoBERTa weight mirror](https://www.modelscope.cn/models/FacebookAI/roberta-base/resolve/master/model.safetensors). MPNet used pinned commit `6996ce1e91bd2a9c7d7f61daec37463394f73f09`, SHA-256 `f8eb85dab52341af018fb8cd325a9d83fdb00871c1b2cb1d445d42b289c1d45e`, and the [ModelScope MPNet weight mirror](https://www.modelscope.cn/models/microsoft/mpnet-base/resolve/master/model.safetensors). The cached bytes had previously been checked against the official Hugging Face file hashes; the helper rehashed each weight for this run. Both models fit all development inputs within 512 tokens, so this ablation compares a truncated first round with the full observed inputs. The small OOF changes give no strong reason to prefer one cap before further paired uncertainty analysis.

I independently checked that both 512-token OOF files contain exactly the same 2,160 IDs, prompt IDs, labels, and fold assignments as E0; neither includes a reserved prompt. Recomputing balanced accuracy from their predictions reproduces the reported values. Local results are in `outputs/e1_frozen_more_512/`; embedding caches and model weights are excluded from Git.
