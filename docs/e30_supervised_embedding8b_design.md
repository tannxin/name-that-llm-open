# E30：Qwen3-Embedding-8B 监督分类分支设计

**状态：预注册脚本与数据预检完成；尚未下载此模型、训练或取得 E30 分数。** 与 [E28 生成基座 QLoRA](e28_qwen8b_lora_feasibility.md)分开编号、分开评估；不得拿 E28 的模型权重或资源数据充当 E30 结果。

## 动机与证据

Qwen 的[官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)列出 `Qwen3-Embedding-8B` 为 Apache-2.0 的 36 层、4096 维文本 embedding 模型，并支持任务指令。[公开 0.680 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484)使用该骨干的监督 LoRA、多深度读出与双头分类作为一个新分支，但没有逐组件消融；它的公开 21 秒预览运行并未测完整隐藏推理。[来源复核](leader_methods_update_2026-11-10.md)

该公开 Notebook 的输入清单含 Kaggle 官方模型资产 `qwen-lm/qwen-3-embedding/Transformers/8b/1`；其训练好的五折 adapter 则来自另一个作者的公开数据集。输入清单中另有 `qwen-lm/qwen-3/Transformers/8b-base/1`，属于生成模型 Base 变体，不能与 embedding 骨干或 E28 锁定的 `Qwen/Qwen3-8B` 混用。[V63 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484)

项目曾测试**冻结**的 Qwen3-Embedding-4B。其效果不能外推到另一个大小的骨干经过监督 LoRA 后的结果；E30 要独立测得单路分数和增量。[冻结模型记录](e1_frozen_embeddings.md)

## 固定比较顺序

1. **权重与资源门槛。** 先锁定官方 Hugging Face 或官方 ModelScope 的不可变 revision，核对配置、tokenizer、权重分片大小与哈希，记录许可证。服务器仅用空闲 RTX 3090 的隔离环境；先完成一个折的无标签前向和一个 epoch 吞吐测试，记录训练峰值显存及外推的五折/全量成本。另用 4,000 条固定无标签文本测完整推理，不以三条公开 Kaggle 样例计时。模型快照不完整、OOM、数值异常或推理预计超出 Kaggle 9 小时时，停止，不换模型或改变折号来补分。
2. **最小监督基线 E30-A。** 只用官方训练集已固定的 120 个开发 prompt 与五折 GroupKFold，输入为原始回答文本，无测试 `prompt_id`、无聊天模板。LoRA 骨干接一个 18 类线性头，从最后非 padding token 读取最后层隐藏状态；分类标签顺序与 E0/E16 一致。先用折 1 检查训练与验证流程，再在资源允许时按同一设置完成五折 OOF。普通交叉熵、一个固定 seed、固定 epoch 和截断长度在首个带标签验证前写入机器配置；不按验证折选择最佳 epoch 或阈值。
3. **只在 E30-A 有互补信号后测试 E30-B。** 保留同一权重快照、数据、折与优化日程，唯一新增少量预先固定层的可训练读出；先检查单路和与 E24 固定融合的净增量，再决定是否加入无需真实测试 prompt_id 的软中心化头。公开方案中的 40% 几何混合权重不能直接照搬到我们的 E24 分数。
4. **再试 E24 外部数据。** 只有官方单源实验与资源检查完成后，才比较 E30-A/B 的官方训练与“官方＋E24 审计行”训练，外部类内总权重沿用 E24 的 0.25，不重新筛回答或按 OOF 改来源。报告来源依赖的长度/标点分布风险，不把外部数据上的开发提升视为独立测试提升。

全五折报告应包含：18 类 balanced accuracy、逐折和逐类召回、与 E24 固定融合的逐行互补性、按 120 个 prompt 配对 bootstrap 的区间、训练与推理资源。任何晋级都要相对**当时最好的固定候选**至少 +0.01、至少 4/5 折不降、无类召回下降 ≥0.05；单折只用于资源和机制检查，不能替代五折晋级判断。公榜仅在离线包通过 4,000 行完整推理及格式检查后做一次外部核验。

## E30-A 预注册实现（尚无训练结果）

代码为 [`run_e30_supervised_embedding8b.py`](../run_e30_supervised_embedding8b.py)，机器配置为 [`configs/e30_qwen3_embedding8b.json`](../configs/e30_qwen3_embedding8b.json)。模型锁定 [Qwen 官方 Hugging Face 提交 `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`](https://huggingface.co/Qwen/Qwen3-Embedding-8B/commit/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af)；官方仓库标记 Apache-2.0，模型权重不并入本项目 MIT 代码仓库。脚本在 GPU 阶段核对四个权重分片和 tokenizer 的官方 SHA-256、config 结构、tokenizer EOS/pad 约定；只使用本地模型目录，不发起下载。E28 的生成模型与这些权重不通用。

先在**任何带标签验证前**运行轻量 `preflight`，将官方训练哈希、E0 manifest/OOF 哈希和机器配置哈希写入 `outputs/e30_qwen3_embedding8b/protocol_lock.json`。2026-11-10 的本地静态检查已确认 2,699 官方行、150 个 prompt、开发集 2,160 行/120 prompt、每折训练 1,728 行/96 prompt、验证 432 行/24 prompt；未加载 E30 模型、未测 E30 准确率。配置锁定 seed 42、256 token、batch 1、累积 8、3 epoch、LoRA r16/alpha32、普通交叉熵，最终 epoch 一次验证；不选择最佳 epoch 或阈值。

模型快照获取须使用上述**完整提交哈希**，保存至本地目录。训练环境按配置固定依赖版本，并在确认显卡空闲后只暴露一张 RTX 3090。资源阶段只训练折 1 的训练侧一个 epoch，随后重新加载 adapter，使用训练侧回答重复构成 4,000 条**无标签代理文本**测试完整推理吞吐。该代理只检验服务器资源，不证明 Kaggle 隐藏测试的耗时。资源门槛为训练峰值 <22 GiB、单 epoch ≤1 小时、外推 3 epoch ≤3 小时、4,000 条推理 ≤3 小时；失败即停止。不能把这一折的结果称为五折提升。

```bash
python run_e30_supervised_embedding8b.py preflight
CUDA_VISIBLE_DEVICES=<idle-3090-index> python run_e30_supervised_embedding8b.py resource --model-dir data/models/qwen3-embedding-8b-1d8ad4c
CUDA_VISIBLE_DEVICES=<idle-3090-index> python run_e30_supervised_embedding8b.py fit-fold --fold 1 --model-dir data/models/qwen3-embedding-8b-1d8ad4c
# 折 1 机制检查通过后，以相同命令和设置运行 --fold 2、3、4、5。
python run_e30_supervised_embedding8b.py summarize
```

每折单独保存 adapter、分类头、完整 18 类 logits、运行元数据和资源用量。`summarize` 只有五折齐全才运行；它先逐行重现 E24 固定融合，再以**预先固定的 1.0 分支权重**加上 E30 行标准化 logits，输出单路 BA、逐折/逐类召回、混淆矩阵、逐行互补数和按 prompt 配对 bootstrap 区间。晋级门槛基于 E24 作为当前固定参照：融合 BA 至少 +0.01、至少 4/5 折不降、任何类召回下降不得达到 0.05。若汇总时已有更好的固定候选，还需相对那个候选重核，不能仅凭对 E24 过线就晋级。
