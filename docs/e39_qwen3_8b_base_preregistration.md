# E39：Qwen3-8B-Base 单骨干 QLoRA（资源预登记）

2026-11-11，在任何 E39 验证分数或 GPU 运行之前固定。本项目 E28 使用 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)；E39 仅将基座换为单独的[预训练 Base 权重](https://huggingface.co/Qwen/Qwen3-8B-Base)，隔离检查骨干差异。公开 [0.680 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484) 将该 Base 模型列为输入，不能把整体公榜分数当作此单路的预期效果。Qwen 模型卡标识 Apache-2.0、8.2B 参数、36 层，Transformers 至少 4.51.0。

## 固定数据、模型与训练日程

- 只用官方 `train.csv` SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`，复用 E0/E16 的 120 prompt 开发集、30 prompt 已用保留集边界及原五折 `GroupKFold`。E39 不读保留集，不用外部回答训练，不按单折指标改超参。
- 基座固定 `Qwen/Qwen3-8B-Base` Hugging Face revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。[固定文件树](https://huggingface.co/Qwen/Qwen3-8B-Base/tree/49e3418fbbbca6ecbdf9608b4d22e5a407081db4)有五片 safetensors，共 16,381,516,776 B；训练前须按该 revision 的 LFS SHA-256 与官方 ModelScope 镜像逐文件比对，并对本地全部运行文件重算 SHA-256。不能用 E28 的 instruct 权重或仅靠模型名称替代文件校验。五片 LFS SHA-256 依次为 `9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b`、`9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f`、`c7dd5a191c8da555def0714550375b3c0117751add408fcdda0c2a79ac0186bc`、`ad8b708792105133e03fe19e2d56c89c709f677f01285f8b3ede8947d59fd9fe`、`fbf24915d47ea030bb68ab0b9488f4515a907185baa6dc26837c9c3f2326a550`。
- 保持 E28 单因素对照：`AutoModelForSequenceClassification` 末尾非 padding token 的 18 类头；回答原文、无指令/聊天模板，最多 256 tokens；NF4 双量化、FP16 计算、LoRA `r=16, alpha=32, dropout=0.05`，目标 q/k/v/o/gate/up/down，分类头 `score` 可训练；seed 42、3 epoch、microbatch 1、梯度累积 8、AdamW `2e-4`、weight decay 0.01、3% warmup 后线性衰减、普通交叉熵、最终 epoch，不做最佳轮选择。软件版本沿用 E28 锁定的 Torch 2.6、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0、bitsandbytes 0.45.5。

## 阶段门槛

1. **下载与 CPU 预检：**服务器当前没有 Base 快照；从 Qwen 官方 ModelScope 镜像获取固定文件，支持断点续传，只在 SHA-256 和大小一致后原子落盘。预检核对官方数据、折索引、依赖、GPU2 UUID、模型文件和空间；GPU2 非空闲则不启动。
2. **GPU 资源测试，不看准确率：**仅折 1 的 1,728 条训练侧回答跑第 1 epoch，另在训练侧文本重复成 4,000 行无标签推理；折 1 验证最多 32 条只做无标签延时。记录包括模型载入的显存峰值、更新成功率、完整推理墙钟。门槛沿用 E28：训练与推理峰值各 <22 GiB，至少 95% 优化器更新成功、无 OOM/NaN，单 epoch ≤60 分钟、3 epoch 外推 ≤3 小时、4,000 行推理 ≤3 小时。外推仅作资源判断，不是完成五折。
3. **性能五折：**资源合格后另行固定脚本与配置并从原始 Base 权重独立训练五折，各折官方训练/验证 prompt 不交叉。必须完整保存 2,160 行、18 类 OOF 和逐折资源。只用已归档 E24 固定融合 `row_z(E24 E0c)+row_z(RoBERTa)+mean_seed row_z(ModernBERT)` 加 **1 倍** `row_z(E39)`，对 E24 的 balanced accuracy 至少 +0.01、至少 4/5 折不降、任何类别召回下降小于 0.05；另报告按 prompt 配对 bootstrap 和对 E28/E30-A 的保守比较。不得依据单折或公榜调权、改轮次或选模型。五折前先独立审查资源报告与协议锁。
4. **部署门槛：**即使五折晋级，也须在 Kaggle 实际 GPU 配额与 Internet off 条件下验证基座、tokenizer、adapter、依赖的离线重载、约 4,000 行完整推理和提交格式；RTX 3090 资源测试不能代替 Kaggle T4×2 兼容性。[Kaggle 公告](https://www.kaggle.com/discussions/product-announcements/735239)称 P100 已于 2026-09-15 停用。数据、模型权重和逐行 OOF 不入 Git。

E39 与 E38 无标签 CPU 检索实验可并行；它们各自使用独立输出目录，不读取对方验证结果来改变配方。
