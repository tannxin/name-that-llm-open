# E41：DeBERTa-v3-large 官方单源五折（资源预登记）

2026-11-11 在任何 E41 GPU 训练或验证分数之前固定。E14 的 `microsoft/deberta-v3-base` 五折单路 balanced accuracy 为 **0.3130**，曾低于同日程 RoBERTa；这不能推出 [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) 的结果。公开 [V63 高分系统](v63_reference_asset_audit_2026-11-11.md)也包含一个 DeBERTa large 核心成员，但没有可用于归因的单路消融。E41 只把 E14 的编码器骨干换成大型版，观察强编码器的独立价值与对 E24 的互补性。

## 固定模型、数据与训练

- 官方数据 `train.csv` SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`，原 E0 `split_manifest.csv` SHA-256 `653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15`，E0 OOF 身份文件 SHA-256 `698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa`。仅 120 个开发 prompt / 2,160 行、五折 GroupKFold；30 个已用保留 prompt 和 Kaggle 测试不参与训练、选模或预检。
- 基座固定 Hugging Face `microsoft/deberta-v3-large` revision `64a8c8eab3e352a784c658aef62be1662607476f`，模型卡许可 **MIT**。该 revision 的 `pytorch_model.bin` 为 **873,673,253 B**，LFS SHA-256 `dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5`；`spm.model` 为 **2,464,616 B**，SHA-256 `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd`。还需固定并重算同 revision 的 `config.json` 与 `tokenizer_config.json`，不得只靠名称或本地缓存别名加载。当前 Pod 未发现该大型版缓存。
- 单因素沿用 [E14](e14_deberta_v3.md)：`AutoModel` 最后一层 attention-mask mean pooling、dropout 0.1、18 类线性头；原文、最多 256 tokens、fast tokenizer；seed 42，固定 12 epoch，batch 16，编码器 LR 2e-5、头 LR 1e-3，AdamW weight decay 0.01，warmup ratio 0.1 后线性衰减、逆训练折类别频数交叉熵、fp16 与成功 optimizer step 才前进的 scheduler。模型与 tokenizer 固定为 Transformers 4.51.3、Torch 2.6.0 系列及 E14 的 SentencePiece 依赖；不看验证 epoch 选轮次。

## 执行与晋级门槛

1. **下载与 CPU 预检：**核验上述固定 revision 的全部运行文件、文件 SHA 和原五折，记录脚本/配置哈希。GPU0 当前 RTX 3090 物理 UUID `GPU-efde52d6-320a-2d05-aa18-13ed62b95259`；启动前必须重新检查占用与共享盘，若 E39 或其他作业使用该卡则等待。允许与 E39 的 GPU2 作业并行，但不在同一 GPU 并发。
2. **GPU 资源测试：**仅折 1 的训练侧 1,728 行跑第 1 epoch；不计算验证准确率或保存验证分数。按原 batch 16、256 tokens 先测；训练 reserved 显存须 **<22 GiB**、无 OOM/NaN、至少 95% optimizer step 成功、单 epoch **≤15 分钟**（12 epoch 外推 ≤3 小时/折）。再用训练侧文本重复成约 4,000 行做无标签推理，须 **≤2 小时**且推理 reserved 显存 **<22 GiB**。若原 batch 16 资源失败，E41 按本次单因素定义停止，不在看到资源或精度后改 batch/梯度累积再次试这一编号。
3. **性能五折：**资源合格且五折脚本和协议锁经独立审核后，五折从同一基座独立训练，固定最终 epoch，保存完整 2,160 行 × 18 类 OOF 与逐折资源。按固定 `row_z(E24 E0c)+row_z(RoBERTa)+mean_seed row_z(ModernBERT)+1×row_z(E41)` 比 E24 **至少 +0.01 BA**、**至少 4/5 折不降**、任何一类召回下降 **小于 0.05**；另报按 prompt 配对 20,000 次 bootstrap。对同折 E14 单路作诊断对照，但不以单折分数修改协议。
4. **部署：**只有五折、逐类和资源均过线后，才全量拟合并在 Kaggle Internet off、实际 GPU/CPU 配额下重载、计时约 4,000 行及检查提交格式。数据、权重、逐行 OOF 不入 Git。

此资源预登记不表示已下载或开始 GPU 训练；训练脚本与配置须在首个性能分数前另行冻结并独立校验。
