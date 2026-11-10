# E1: frozen embedding comparisons

`run_frozen_embeddings.py` uses the fixed E0 manifest and checks every `id`,
`prompt_id`, label, and five-fold assignment against E0's OOF file before fitting.
It embeds only the 2,160 development rows from 120 prompts. The 30 holdout
prompts are never encoded or scored. Every fold fits the same balanced linear
SVM (`C=1`) as E0 on its training prompts, then predicts its validation prompts.

The first comparison gives both encoders a 256-token cap. The
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
specifies a 256-token default; the
[E5 model card](https://huggingface.co/intfloat/e5-base-v2) permits 512 tokens
and recommends `query: ` for non-retrieval tasks. Both use attention-masked
mean pooling followed by L2 normalization. A 512-token E5 run is a separate
length ablation and must use another output directory.

## Reproduce

On the experiment server, the project storage does not support the symlink
that Python's `venv` creates, and `ensurepip` is absent. An isolated
environment was placed under the user's home directory instead:

```bash
python3 -m venv --without-pip ~/namethatllm-venv
python3 -m pip --python ~/namethatllm-venv install pip
~/namethatllm-venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
~/namethatllm-venv/bin/python -m pip install -r requirements.txt transformers==4.51.3
```

The server uses NVIDIA driver 535.154.05 (CUDA 12.2). Its globally installed
PyTorch 2.12.0+cu130 cannot use CUDA on that driver; the project environment's
PyTorch 2.6.0+cu118 can. To rerun one encoder at the same commit:

```bash
CUDA_VISIBLE_DEVICES=2 ~/namethatllm-venv/bin/python run_frozen_embeddings.py \
  --models all-minilm-l6-v2 \
  --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --device cuda:0
```

The exact revision, weight-file SHA-256, input cap, truncation count, package
versions, fold scores, and pooled OOF score are saved in each model's
`metrics.json`. A cached embedding file is reused only when its revision,
dataset hash, row IDs, pooling, prefix, and token cap match.

## First result

| Encoder | Commit | Cap | Truncated development rows | Pooled OOF balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| TF-IDF + SVM E0 | n/a | full text | n/a | 0.281019 |
| all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | 256 | 265 / 2160 | 0.128704 |
| e5-base-v2 | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | 256 | 276 / 2160 | 0.184722 |
| e5-base-v2 (length ablation) | same commit | 512 | 0 / 2160 | 0.196759 |
| Qwen3-Embedding-0.6B, task instruction | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 256 | 310 / 2160 | 0.177778 |
| Qwen3-Embedding-0.6B, no instruction (ablation) | same commit | 256 | 239 / 2160 | 0.162037 |
| BGE-M3 | `5617a9f61b028005a4858fdac845db406aefb181` | 256 | 518 / 2160 | 0.151852 |
| BGE-M3 (length ablation) | same commit | 512 | 0 / 2160 | 0.159722 |
| Qwen3-Embedding-4B, task instruction | `5cf2132abc99cad020ac570b19d031efec650f2b` | 256 | 310 / 2160 | 0.200463 |
| Qwen3-Embedding-4B (length ablation) | same commit | 512 | 0 / 2160 | 0.190741 |

The MiniLM run used the server's CPU PyTorch 2.12.0+cu130 for embedding
computation. It produced the same score when the cached embeddings were
reevaluated. The model's `model.safetensors` SHA-256 is
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`,
matching the Hub blob identifier.

The E5 run used the project environment's PyTorch 2.6.0+cu118 on physical GPU
2. Its `model.safetensors` SHA-256 is
`d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693`.
The Hugging Face transfer repeatedly stalled, so the exact weight file was
resumed from the [model author's ModelScope mirror](https://www.modelscope.cn/models/intfloat/e5-base-v2).
ModelScope's `X-Linked-Etag` matched the Hub SHA-256, and the completed file
was independently hashed before it was used. The resolved Hub snapshot commit
and final hash are saved in `metrics.json`.

The E5 512-token run used a separate `outputs/e1_frozen_e5_512` directory.
It recovered 1.20 percentage points of balanced accuracy relative to the
shared 256-token cap; none of the development rows exceeded 512 tokens.

The Qwen 4B 512-token run changed only the length cap; it removed all
development truncation but scored below its 256-token counterpart. Its fixed
equal-weight fusion with E0c and fine-tuned RoBERTa also remained below E10.
See [the length comparison](e1_qwen4b_length.md).

The [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
recommends a task-specific English instruction. The first run prepended
`Instruct: Identify which language model generated the following response.\nQuery:`
to each answer, costing 14 tokens. It used left padding, final-token pooling,
L2 normalization, and bfloat16 on physical GPU 2. The separate
`outputs/e1_frozen_qwen_no_instruction` run omitted the prefix; its score was
1.57 percentage points lower. The model weight SHA-256 was
`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`
in both runs. The file was obtained from the
[model author's ModelScope mirror](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)
and checked against the Hub's content hash before inference.

The [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) says no instruction
prefix is needed. Its [pooling configuration](https://huggingface.co/BAAI/bge-m3/blob/main/1_Pooling/config.json)
and the [authors' inference documentation](https://github.com/FlagOpen/FlagEmbedding/blob/master/docs/source/bge/bge_m3.rst)
specify the normalized CLS hidden state for dense vectors. The shared 256-token
cap truncated 518 development rows. Its 2,271,145,830-byte `pytorch_model.bin`
was downloaded from the [author's ModelScope mirror](https://www.modelscope.cn/models/BAAI/bge-m3)
and verified against the official SHA-256
`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`.
The tokenizer and configuration files were checked against the pinned Hub
revision's Git or LFS hashes before GPU 2 inference.
The separate `outputs/e1_frozen_bge_512` run removed all truncation and gained
0.79 percentage points of balanced accuracy. It used 2.47 GB peak allocated
CUDA memory and 30.84 seconds for development-row embedding inference at batch
size 8, excluding the 4.73-second model load.

The [Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
lists an Apache-2.0 license and the same instruction format, left padding, and
last-token pooling as the 0.6B model. Its two safetensors shards were downloaded
from the [author's ModelScope mirror](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B)
and independently checked against the pinned Hub LFS SHA-256 values:
`e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8`
(4,965,826,464 bytes) and
`ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1`
(3,077,765,624 bytes). On physical GPU 2 with bfloat16, SDPA attention, and
batch size 4, model loading took 4.79 seconds and development-row embedding
inference took 75.70 seconds. Peak CUDA allocation was 8.37 GB (7.80 GiB).
Its pooled OOF score exceeded the 0.6B variant by 2.27 percentage points.

All five frozen encoders tested so far score below the E0 lexical baseline on
this grouped OOF evaluation. These are frozen representations with a linear
head; the result does not assess supervised encoder fine-tuning.
