# Embedding 候选模型核对（2026-11-04）

历史对话中使用了若干简称。实验时须记录具体仓库与 revision；冻结表示采用模型卡推荐的输入格式和 pooling。后续微调结果需把全参数更新与 LoRA/QLoRA 分开标记。

| 历史简称 | 实际模型仓库 | 编码要点 | 许可证/访问备注 |
| --- | --- | --- | --- |
| all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Mean pooling，384 维，常用长度上限 256 | Apache-2.0 |
| e5-base-v2 | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | Mean pooling，768 维；对称分类使用 `query: ` 前缀 | MIT |
| BGE-M3 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | CLS，1024 维，支持长文本 | MIT |
| mpnet-base | [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) | 原始 MLM 编码器，需事先固定 pooling 规则 | 模型仓库未标明许可证；开源仓库不复制权重 |
| all-mpnet-base-v2 | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Mean pooling，768 维 | Apache-2.0 |
| roberta-base | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | 原始 MLM 编码器，需事先固定 pooling 规则 | MIT |
| Style-Embedding | [AnnaWegmann/Style-Embedding](https://huggingface.co/AnnaWegmann/Style-Embedding) | Mean pooling，768 维 | HF 权重仓库未标明许可证；[上游代码](https://github.com/nlpsoc/Style-Embeddings)为 MIT，权重不复制到开源仓库 |
| Qwen3-Embedding-0.6B | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 左填充、最后 token pooling，1024 维 | Apache-2.0 |
| Qwen3-Embedding-4B | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 左填充、最后 token pooling，2560 维 | Apache-2.0 |
| Qwen3-Embedding-8B | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 左填充、最后 token pooling，4096 维 | Apache-2.0 |
| LLM2Vec-Llama-3-8B | [McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised) | Mean pooling；须组合 adapter 与 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 基座 | 基座需另行获取授权；不得把适配器的 MIT 许可误当作基座许可 |

Qwen 的检索 query 指令与当前回答来源分类任务不同。首轮采用模型卡建议的**任务专属英文分类指令**，并把无指令输入作为单独消融；不使用检索任务描述。模型精度、批量大小和最大长度在相同开发集折上统一记录；最终还要验证 Kaggle 断网与 9 小时推理限制。
