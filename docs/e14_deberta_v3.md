# E14: DeBERTa-v3-base supervised encoder

## Protocol

This is a single new-backbone comparison against the corrected RoBERTa E2d-R run. The only planned change is the pretrained encoder: `microsoft/deberta-v3-base` at exact Hugging Face revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. The five prompt-grouped development folds, seed 42, 12 fixed epochs, 256-token limit, batch size 16, mean pooling, class-weighted 18-way cross entropy, head and encoder learning rates, warmup, weight decay, dropout, mixed precision, and optimizer-step-aware scheduler are the same as E2d-R. The training, split-manifest, and E0 OOF SHA-256 hashes match E2d-R exactly. No validation epoch was selected. The locked 30-prompt holdout and Kaggle test were not read.

The Pod used Python 3.10.12, PyTorch 2.6.0+cu118, Transformers 4.51.3, `sentencepiece==0.2.1`, and `protobuf==6.33.0` on physical GPU 2 (RTX 3090). `HF_HUB_DISABLE_XET=1` was needed after the Xet download stalled; the completed run verified that the loaded pretrained revision equals the requested SHA. The DeBERTa-v2 fast tokenizer emitted Transformers' byte-fallback warning, so rare characters may be tokenized differently from the slow SentencePiece tokenizer. Both fitting and OOF inference used the same fast tokenizer. No model checkpoints were saved, so this run is an OOF comparison rather than an inference package.

## Development OOF results

Each class contributes 120 development rows, so the pooled 18-class balanced accuracy equals ordinary accuracy here. The saved score files were checked for exactly matching `id`, `prompt_id`, `true_model`, and `fold` columns before fusion. The fixed E10 score rule was reproduced exactly from its saved predictions.

| Model or fixed score fusion | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Pooled balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2d-R RoBERTa | 0.3194 | 0.2894 | 0.3611 | 0.3495 | 0.3495 | 0.3338 |
| E14 DeBERTa | 0.2870 | 0.3148 | 0.3333 | 0.3148 | 0.3148 | **0.3130** |
| E10 E0c + RoBERTa | 0.3681 | 0.4028 | 0.4491 | 0.3727 | 0.4051 | **0.3995** |
| E0c + DeBERTa, two equal row-z streams | 0.3750 | 0.3958 | 0.4282 | 0.3819 | 0.3727 | 0.3907 |
| E0c + RoBERTa + DeBERTa, three equal row-z streams | 0.3634 | 0.3981 | 0.4352 | 0.3866 | 0.4005 | 0.3968 |

DeBERTa versus RoBERTa differs by **-0.0208** balanced accuracy. A paired bootstrap over 120 prompts (20,000 draws, seed 42) gives a 95% percentile interval of **[-0.0417, 0.0000]**. The fixed three-stream fusion differs from E10 by **-0.0028**, interval **[-0.0167, +0.0106]**. Replacing RoBERTa with DeBERTa in the two-stream E10 rule gives **-0.0088**, interval **[-0.0255, +0.0079]**. These intervals are exploratory and do not adjust for other project experiments on the same development prompts.

The DeBERTa leaf predictions are correct on 211 rows that E10 misses, but wrong on 398 that E10 gets right. Family accuracy is 0.7120 for DeBERTa, 0.7157 for RoBERTa, and 0.7560 for E10. Adding DeBERTa raises E10 family accuracy slightly to 0.7583 while lowering the required 18-class score to 0.3968. This run does not justify replacing or extending the submitted E10 pipeline.

## Reproduction

From the server project root, with the same data and split files present, install the tokenizer dependencies in the experiment environment and run in a **new empty output directory**:

```bash
/root/namethatllm-e1-venv2/bin/python -m pip install sentencepiece==0.2.1 protobuf==6.33.0
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2 \
  /root/namethatllm-e1-venv2/bin/python run_finetune_encoder.py \
  --model-id microsoft/deberta-v3-base \
  --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --epochs 12 --pooling mean --max-length 256 --batch-size 16 \
  --no-save-checkpoints --output-dir outputs/e14_deberta_v3_base_12ep_256_reproduce
```

The completed metrics and OOF predictions are in the Git-ignored `outputs/e14_deberta_v3_base_12ep_256_http/` directory on the Pod and in the local workspace. The score comparison uses the existing `row_standardize` and `paired_prompt_bootstrap` functions in `run_score_normalization.py`, with equal weight for each included score stream.
