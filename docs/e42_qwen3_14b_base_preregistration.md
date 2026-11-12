# E42：Qwen3-14B-Base 单骨干 QLoRA（预登记）

2026-11-12，在下载模型、GPU 资源测试及任何 E42 验证分数之前固定。E39 的五折目前被共享 GPU2 的其他作业阻挡；E42 是预先固定的独立候选，不根据 E39 分数提出。E42 仅将 E39 的 Qwen3-8B-Base 换成 [Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)，检验**同系列更大预训练骨干**是否带来新的模型来源线索。两者都不使用 Instruct/聊天权重，除此之外训练、数据、融合与晋级规则保持一致。8B 与 14B 的层数、宽度等结构也不同，因此结果只说明整个骨干替换的作用，不能单独归因于参数量。

## 固定输入和权重

- 只使用比赛官方 `train.csv`，SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`。使用原 E0/E16 分割清单（SHA-256 `653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15`）：120 个开发 prompt、2160 行、按 prompt 分组的原五折，每折训练 96 个 prompt/1728 行，验证 24 个 prompt/432 行。30 个已用保留 prompt 始终隔离，不读取、不训练、不选择参数。不添加外部回答。
- 固定 Hugging Face `Qwen/Qwen3-14B-Base` revision [`0b0bd3732e2c374d483664439ea334928b65f304`](https://huggingface.co/Qwen/Qwen3-14B-Base/tree/0b0bd3732e2c374d483664439ea334928b65f304)，模型卡许可证 Apache-2.0。官方[原始上传提交](https://huggingface.co/Qwen/Qwen3-14B-Base/commit/ec06e99f90d109fdac1e2525a826de4e7e62ff83)列出 8 个 BF16 safetensors 分片的 LFS SHA-256/字节数；见下表。八片文件合计 29,536,665,640 B。2026-11-12 只读比对 HF 固定树与 Qwen ModelScope 镜像，15 个运行文件的大小和 SHA-256 相同，小文件另由固定 HF Git blob ID 约束；下载器将再次核对。索引中的 `metadata.total_size=29,549,721,600` 比实际八片文件总大小多 13,055,960 B，须在 CPU 预检标注，不得误当文件完整性失败或忽略真实分片校验。任何模型文件不符直接停止。

| 分片编号 | 字节数 | SHA-256 |
| --- | ---: | --- |
| 1 | 3,841,788,544 | `fd1c6d2cf56f0a67d4d1491e8fee36cc6e5bfd6a114ab4e61667a63763662b3a` |
| 2 | 3,963,750,816 | `92dbde5d38ecb4fb25b405d1c4d46907481317be6646e8492a22463b48fd09a5` |
| 3 | 3,963,750,880 | `60945918edc6fb2eca950ca2d6d350020f31b805d02b7cac53cd1fac96c1a804` |
| 4 | 3,963,750,880 | `dc3757a85266e177e57719b334052137d881f4d5a586a4a239c5844c00f9fbb5` |
| 5 | 3,963,750,880 | `10d1be9a383fd3371f00916d26484a263c08649bfc59ba94a7afe78ae86285ce` |
| 6 | 3,963,750,880 | `c046e2c218ad550b8a535ce001b258913c5d956e243d01c2a67acc8c4bca8294` |
| 7 | 3,963,750,880 | `27a80e8c324120762187ae76b8cb4983e332b6843645d79137cefe964b19ed2a` |
| 8 | 1,912,371,880 | `73bcc7f129c1884540a8ea079cd5e8b3ec07f4cd66bf1b5a71d3d2c276af160f` |

## 固定训练和资源协议

- 沿用 E39：`AutoModelForSequenceClassification` 末尾非 padding token 的 18 类头；回答原文，无聊天模板或额外指令；最长 256 tokens。NF4 双量化、FP16 计算；LoRA `r=16, alpha=32, dropout=0.05`，目标 q/k/v/o/gate/up/down，分类头 `score` 可训练。seed 42、3 epochs、microbatch 1、梯度累积 8、AdamW `2e-4`、weight decay 0.01、3% warmup 后线性衰减、普通交叉熵、最终 epoch；不按验证分数选轮次或权重。Torch 2.6.0、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0、bitsandbytes 0.45.5。绝不加载 E39/E28 的 adapter。
- 先在 Pod 本地临时盘断点续传，使用固定 revision 的官方文件大小、SHA-256 和小文件 Git blob ID 校验；再原子导入持久模型目录。逐文件哈希及导入报告必须完成。下载和 CPU 预检均不加载 GPU，也不计算验证分数。
- GPU 资源测试仅用**物理 GPU1**，要求 Pod 稳定、该卡持续空闲、原工作不被中断、模型和数据哈希通过。只用折 1 训练侧 1728 行训练第 1 epoch，训练侧文本重复成 4000 行无标签推理，保存训练/加载/推理耗时、显存峰值和成功优化更新数；不能读取折 1 验证标签或准确率。资源门槛沿用 E39：训练和推理峰值各 **<22 GiB**，成功优化更新 **≥95%**，无 OOM/NaN，首 epoch **≤60 分钟**，三轮线性外推 **≤3 小时**，4000 行推理 **≤3 小时**。任何一项失败，E42 停止；不能因为单折结果改配方。
- 资源通过后另行审查五折训练脚本和部署预算，才可执行性能五折。五折需分别从原始 14B Base 初始化、保存 2160 唯一 OOF 行、逐折逐类召回与资源；训练/验证 prompt 不交叉。固定融合为已归档 E24 的 `row_z(E0c)+row_z(RoBERTa)+mean_seed row_z(ModernBERT)` **加 1 倍** `row_z(E42 logits)`。相对 E24 的 balanced accuracy 增量 **≥0.01**、至少 **4/5** 折不降、任一类召回下降 **<0.05**，三项同时满足才晋级；另报告 20,000 次 seed 42 prompt 配对 bootstrap 及 E28/E30-A/E39 的保守比较，不据它们调权。
- Kaggle 部署是独立门槛：需在实际分配的 GPU 上实测离线重载、完整约 4000 行推理、格式和 9 小时总体预算；预计 T4×2 配置须单独测量，不能假定每次都分配该配置。14B 的 BF16 权重约 29.5 GB，不能直接装入单张 16 GB T4；NF4 在 RTX 3090 上的成功也不能证明 T4 可行。未通过真实部署前不训练全量 adapter、不提交 Kaggle。数据、权重、逐行 OOF 不入 Git。
