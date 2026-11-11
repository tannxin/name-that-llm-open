# E30-B：Qwen3-Embedding-8B 三深度读出单因素实验

**预登记于 2026-11-11（E30-B 任何带标签验证前）。状态：本地 CPU 协议预检已完成，尚未启动 E30-B GPU 资源试验或五折训练。** 这是 [E30 总设计](e30_supervised_embedding8b_design.md)中 E30-A 有互补信号后的下一项消融。E30-A 五折已有结果，但本实验的层数与层位按 36 层模型结构等距固定，未根据 E30-A 的逐类召回、逐行 OOF 或 E36 的误差挑选。

## 唯一改变与固定先验

[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)说明骨干有 36 层、隐藏宽度 4096。[公开比赛方案来源审计](leader_methods_update_2026-11-10.md)说明多深度读出属于可测试候选，但公开 Notebook 将它与其他训练和融合组件一起使用，没有可归因的逐组件提升。因此 E30-B 只测试读出结构。

E30-A 从最后一层的末位非 padding token 取 4096 维，接一个 18 类线性头。E30-B 从隐藏状态索引 **12、24、36** 取同一 token 的 3 个 4096 维向量；索引 0 是 embedding，36 是最后一层。按此顺序拼接为 12288 维，接**一个** 18 类线性头。无层权重搜索、层注意力、归一化、双头、中心化头或附加损失。选择 12/24/36 是把 36 层等分为三段的结构先验，后续不能看折分数改变层号。

输入仍为原始官方回答文本，由同一固定 tokenizer 自动添加特殊 token，右侧 padding，最多 256 token；取 attention mask 指向的最后有效 token。官方模型 revision、四个权重分片的哈希核对、NF4 量化、LoRA r16/alpha32/dropout0.05 与七类投影目标、seed42、五折 GroupKFold、120 个开发 prompt、3 epoch、batch1/累积8、AdamW 2e-4/weight decay 0.01、线性 warmup 3%、普通交叉熵和仅最终 epoch 验证全部沿用 E30-A。预检会读取完整官方训练文件以核对哈希和固定分区，但保留的 30 个 prompt 不参与训练、验证或选模；不读取隐藏测试回答，也不按验证选择 epoch、阈值或融合系数。E30-A 的代码和配置不改动；E30-B 通过校验其 SHA-256 后复用模型加载、数据切分、优化器与 tokenizer 函数。[机器配置](../configs/e30b_qwen3_embedding8b_multidepth.json) · [运行脚本](../run_e30b_multidepth.py)

## 预先固定的评估

五折必须各有 432 个唯一官方开发 ID、24 个互斥验证 prompt、18 类有限 logits；拼接后精确覆盖 2160 行/120 prompt，且与 E0 的行 ID、真值、折号逐行一致。单路 E30-A 与 E30-B balanced accuracy、逐折和逐类召回，以及 E30-B 单路的 18×18 混淆矩阵为描述性结果。主决策只比较两个固定融合：

`A = argmax(E24_fixed_scores + 1.0 × row_z(E30-A logits))`

`B = argmax(E24_fixed_scores + 1.0 × row_z(E30-B logits))`

`E24_fixed_scores` 必须从 E24 审计固定的 E0c、RoBERTa 和三 seed ModernBERT 分数逐行重建，且 2160 个预测须与归档 E24 OOF 完全一致；E30-A OOF 必须匹配其独立审计 SHA-256。融合权重恒为 1.0，不能根据 E30-B OOF 调整。对 B−A 计算 120 个 prompt 为抽样单位的配对 bootstrap（20,000 次，seed42）、逐折差、逐类召回差、错→对与对→错。

**结构晋级门槛**（同时满足）：B 相对 A 的合并 balanced accuracy 至少 **+0.01**；至少 **4/5** 折不下降；每类召回下降都必须严格小于 **0.05**。等于 −0.05 也失败。E30-A 已知的固定融合 0.448611 是预设主比较器；E28 虽有较低 BA，但其未晋级结果及 E36 失败结果都不用于选择 B 的层位或调权。B 通过本门槛仅表示多深度结构在开发集上胜过 A；是否部署还须审核与其他锁定候选的逐类关系、全量 adapter、完整离线包的资源和格式。不得把 OOF 当作 Kaggle 公榜分数。

## CPU 锁与资源门槛

先运行 `preflight`：核对官方训练/E0 manifest/E0 OOF、E30-A 脚本/配置/本机协议锁的 SHA-256，验证开发集切分尺寸，做不加载模型的三层索引与拼接 CPU 契约检查，再以只创建方式写出 `outputs/e30b_qwen3_embedding8b_v1/protocol_lock.json`。这个目录被 Git 忽略；锁内明确 `validation_accuracy_computed=false`、`gpu_used=false`。本地锁生成时间为 **2026-11-11 00:41:18 UTC**；本地 B 配置 SHA-256 为 `b46f82add551e7440c4fc2dd4ae9fccfec5aa9ea627df0fc2fbe68d7e4ce5d09`，B 脚本 SHA-256 为 `d660aa5e70cb25066cdecdd1e9425e8328cbdc68b655753e6f75443433f8895b`。服务器须从最终相同字节脚本和配置另外运行自己的 CPU `preflight`，确认哈希一致后才允许 GPU；本地和服务器的协议锁文件可因生成时间和本机 A 锁字节而不同。

资源试验只用第 1 折训练侧 1728 行训练 **1 epoch**，然后重载 adapter 与分类头，对训练侧回答循环组成的 **4000 条无标签代理文本**完整推理。资源阶段不计算验证准确率。与 E30-A 同一硬门槛：RTX 3090 物理 UUID 空闲且 PyTorch 只见该卡；训练及推理峰值保留显存各 <22 GiB；至少 95% 计划优化器更新完成；单 epoch ≤3600 秒、外推 3 epoch ≤10800 秒、单 adapter 4000 行推理 ≤10800 秒，logits 全有限。任一失败停止，不降低层数、截断长度或 batch 来挽救该版本。资源通过后才按锁运行折 1–5，最后一次性汇总五折，不据单折标签分数调参。

```bash
python run_e30b_multidepth.py preflight
CUDA_VISIBLE_DEVICES=<idle-3090-GPU-UUID> python run_e30b_multidepth.py resource --model-dir data/models/qwen3-embedding-8b-1d8ad4c
CUDA_VISIBLE_DEVICES=<idle-3090-GPU-UUID> python run_e30b_multidepth.py fit-fold --fold 1 --model-dir data/models/qwen3-embedding-8b-1d8ad4c
# 使用相同配置顺序运行 --fold 2、3、4、5；GPU 空闲检查在每折开始前重复。
python run_e30b_multidepth.py summarize
```

所有 GPU 命令须由项目根目录、E30-A 相同固定 Python 依赖环境执行，使用同一已核验的模型目录。服务器上实际模型路径和 GPU UUID 应在执行时从已审计环境读取，不能靠文档占位符猜测。E30-B 五折的 adapter、分类头和逐行 OOF 只保存在服务器及 Git 忽略目录；公开仓库仅同步代码、机器配置、协议、资源/效果的聚合审计，不同步官方训练、测试回答、模型权重、逐行预测或内部挂载路径。
