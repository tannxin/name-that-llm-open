# Original 0.684 notebook model attachments

The 64-text private diagnostic completed feature extraction with Qwen3-8B-Base. It stopped at the next decoder after 2,540 seconds because its inputs contained no Qwen2 checkpoint of hidden size 3,584. The final ensemble did not run. The original author's public score is 0.684. These private runs use unlabeled text and have no leaderboard score. [Original Kaggle notebook](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline)

Kaggle's model catalogue resolves all seven numeric model-version IDs embedded in the author's Version 1 notebook. The checked 34-input private environment contained six of those versions.

| Original model | Kaggle model version ID | Present among the 34 inputs |
|---|---:|---|
| Qwen2.5-7B | 166254 | No |
| Qwen2.5-14B | 166265 | Yes |
| Qwen3-4B-Base | 363135 | Yes |
| Qwen3-8B | 363139 | Yes |
| Qwen3-8B-Base | 363140 | Yes |
| Qwen3-14B | 363149 | Yes |
| Qwen3-14B-Base | 363153 | Yes |

The Qwen2.5 LoRA training script names a 28-layer backbone with hidden size 3,584 and keeps 20 layers. Its only model attachment and its published adapter configuration both identify QwenLM's Transformers 7b Version 1. The 7b-instruct variation has a different ID, 166258. [Training script](https://www.kaggle.com/code/vnton37nguyn/ntl-train-lora-qwen25-instyle) · [Official Kaggle model](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/7b/1)

The public, seed-7 and Claude Qwen3 LoRA adapters all declare Qwen3-8B-Base Version 1. The original lookup uses model type and hidden size, which match five configurations in the checked environment. Two lack complete weights, and the other three belong to distinct model variants. The declared base resolves this ambiguity for the adapters. The Claude dataset stores its completion marker directly under `final/`, which the author's recursive search supports. [Claude adapter dataset](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) · [Official 8B Base](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/8b-base/1)

The original code also calls Llama-3.1-8B and Qwen3-Embedding-8B. Their Kaggle model IDs in the checked environment are 1041713 and 426337 respectively. Both appear in the author's downloadable metadata, outside the seven IDs embedded in the notebook. The final DeBERTa encoder and head can be traced to dataset version 21119693. The component stack looks up a separate DeBERTa directory. No original source cell references Gemma. [Llama checkpoint](https://www.kaggle.com/models/alexchilton/llama-3-1-8b-base/Transformers/default/1) · [Embedding checkpoint](https://www.kaggle.com/models/qwen-lm/qwen-3-embedding/Transformers/8b/1) · [DeBERTa assets](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend/versions/1)

## What the 35-input CPU run verified

The private CPU run found all 35 expected attachments and checked 47 component routes. All four Qwen2.5-7B shards, totaling 15,231,271,888 bytes, matched the publisher's SHA-256 digests. Its configuration, index and four tokenizer files also matched. The same run verified the five Qwen3-8B-Base shards. It found the Claude adapter through the author's recursive lookup and confirmed the base declared by all three Qwen3 adapters. [Pinned upstream 7B files](https://huggingface.co/Qwen/Qwen2.5-7B/tree/d149729398750b98c0af14eb82c78cfe92750796)

The GSQ check confirmed all nine file sizes specified by the author. Of 2,003 index entries, 15 referred to missing tensors. All 15 began with `mtp.`, the exact exception allowed by the original loader. These CPU results establish the checked files and structures. Historical branch execution and contribution to the leaderboard score remain unverified.

**The 35-input check reported an Open Pool mismatch.** Its receipt recorded nine file-size differences and a changed directory listing. We compared that exact pattern with Kaggle’s file lists for all 18 published versions. Version 18 alone matched. It lists 32 files and a 115,325,974-byte `ntl_open_pool.parquet`. The earlier private replay used Version 17 (Kaggle dataset version ID 22123604), with 30 files and a 112,851,698-byte Parquet. These sizes come from the public file lists. The check recorded the mismatch without recording the observed byte counts. The historical author's Open Pool bytes remain unresolved. [Version 17](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/17) · [Version 18](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/18)

After the Open Pool input was pinned to Version 17, the same private CPU check completed in **510.5 seconds**. The strict 35-input inventory and the 30-file Open Pool check passed, and the 47-route audit recorded zero errors and zero warnings. The measured Qwen checkpoint results and the original-loader checks for Claude and GSQ matched the previous run. The revised 64-text GPU diagnostic now checks the Open Pool file list and sizes before reading its data. CPU verification is complete. A successful end-to-end replay or a new leaderboard score still requires separate evidence.
