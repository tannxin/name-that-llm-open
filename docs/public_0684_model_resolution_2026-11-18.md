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

The GSQ loader explicitly permits stale index entries starting with `mtp.`. A generic index check flagged this model without recording which keys were absent. The revised CPU check applies the original exception across every indexed tensor and also verifies nine file sizes fixed in the source. Its separate 7B integrity check compares four mounted weight shards, totaling 15,231,271,888 bytes, with the publisher's SHA-256 digests. Attachment identity and integrity checks alone cannot establish historical branch execution or score contribution. [Pinned upstream 7B files](https://huggingface.co/Qwen/Qwen2.5-7B/tree/d149729398750b98c0af14eb82c78cfe92750796)
