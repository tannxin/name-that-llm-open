# Qwen3-Embedding-4B 512-token frozen ablation

The original frozen Qwen3-Embedding-4B experiment used a 256-token cap and truncated 310 of the 2,160 development answers. This comparison changes only the cap to 512 tokens, with the same pinned checkpoint (`5cf2132abc99cad020ac570b19d031efec650f2b`), task instruction, last-token pooling, balanced linear SVM head, five prompt-grouped folds, and 120 development prompts. The 30-prompt reserved holdout was not encoded or scored. The 512-token run truncated no development answer.

| Setting | Qwen-only pooled OOF balanced accuracy | Equal-weight row-z-score fusion with E0c and corrected RoBERTa |
| --- | ---: | ---: |
| Qwen 4B, 256 tokens | 0.200463 | 0.382407 |
| Qwen 4B, 512 tokens | 0.190741 | 0.385185 |
| E10, no Qwen | — | **0.399537** |

The Qwen-only 512-minus-256 difference is -0.009722, with a prompt-paired bootstrap 95% interval of [-0.018519, -0.000926]. The fixed three-stream 512-token fusion is 0.014352 below E10, with a prompt-paired interval of [-0.029167, 0.000463]. These are exploratory development comparisons and do not adjust for the project's previous model searches. Neither Qwen variant improved the existing E10 fusion under this fixed equal-weight rule. No new Kaggle submission was made.

The server run used physical GPU 2, batch size 2, PyTorch 2.6.0+cu118, and the existing pinned Hugging Face snapshot. Development embedding inference took 80.4 seconds. Full metrics, OOF predictions, decision scores, and embedding cache are in Git-ignored `outputs/e1_frozen_qwen4b_512/qwen3-embedding-4b/`. The result tests only frozen embeddings with a linear head; it does not assess supervised Qwen fine-tuning.

Reproduce with the existing project environment on the experiment server:

```bash
CUDA_VISIBLE_DEVICES=2 python run_frozen_embeddings.py \
  --models qwen3-embedding-4b \
  --model-revision 5cf2132abc99cad020ac570b19d031efec650f2b \
  --qwen-max-length 512 --batch-size 2 --device cuda:0 \
  --output-dir outputs/e1_frozen_qwen4b_512
```
