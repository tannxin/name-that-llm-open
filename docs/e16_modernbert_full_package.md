# E16：ModernBERT 三种子全量训练与离线推理包

**状态：已完成。** 三份全量训练包均通过独立 CPU 重载、预测一致性和本地逐文件哈希核验；Kaggle Notebook 打包与外部验证由后续流程执行。

## 目的与数据边界

[E16 五折随机种子复核](e16_modernbert_seeds.md)预先指定 seed 42、17、73 的 ModernBERT 分数平均为第三路，并在开发集上达到 balanced accuracy 0.409259，较 E10 提高 0.009722；按 prompt 配对的区间仍跨零。这里仅把这三路 ModernBERT 重新拟合在**官方完整训练集**上，供一次 Kaggle 外部验证使用。全部 150 个 prompt、2,699 条实际存在的回答均进入训练，其中 `claude-fable-5` 有 149 条，其余 17 类各 150 条。原数据有一条缺失，未补造。原始 `data/train.csv` 的 SHA-256 为 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`；加载前检查行数、prompt 数、ID 唯一性及 E0 标签顺序。

原来 30 个锁定保留 prompt 现已进入全量训练，因此本阶段不产生新的本地泛化分数，也不把训练损失解释为准确率。只有先前的 120-prompt 五折 OOF 是方案选择依据；后续公榜可用于一次外部验证，不能据此反复调权重或删 seed。

## 固定拟合配方

独立脚本 [fit_modernbert_full.py](../fit_modernbert_full.py)固定 `answerdotai/ModernBERT-base` revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`。与 E15/E16 开发实验一致：末层 token 隐状态按 attention mask 求均值、dropout 0.1、18 类线性头；256 token、batch 16、AdamW、encoder/head 学习率分别 `2e-5`/`1e-3`、weight decay 0.01、10% linear warmup、逆类频率交叉熵、fp16 autocast/GradScaler、12 个固定 epoch，最后一轮直接保存，不用训练集或旧保留集挑 epoch。脚本在开始前对 E15 原配置逐项校验。三个 seed 各自从同一固定预训练快照开始、使用同一份 2,699 行数据，在服务器物理 GPU2 上顺序运行，各用独立输出目录。

推理函数 `load_package(path, device='cpu')` 离线加载并校验脚本、版本、模型、标签与全部包内文件 SHA-256；`score_texts(model, tokenizer, texts, batch_size=16, device='cpu')` 返回官方顺序的 `float32 [N,18]` **原始 logits**。Notebook 应先对每个 seed 的 18 个 logit 逐行标准化，再平均三个 ModernBERT seed；然后与 E0c、RoBERTa 两路各等权相加，完全沿用预定 E16 规则。

每个包含 `modernbert_state_dict.pt`、`encoder_config/config.json`、`tokenizer/`、`smoke_reference.json`、`metadata.json` 和 `smoke_verification.json`。`metadata.artifact_sha256` 覆盖权重、config、tokenizer 文件及 smoke reference；`metadata.source_sha256['fit_modernbert_full.py']` 记录完整拟合时脚本哈希。meta 同时记录训练数据、预训练快照、E15 配置哈希、类计数、训练参数、运行环境、每轮训练损失和跳过的优化器步数。训练数据和大权重包继续保存在 Git 忽略的服务器 `outputs/` 下，Git 仅追踪脚本及本审计文档。

保存前在 CPU 上对三个固定合成文本连续打分两次；重载后再连续打分两次，检查结果与保存前的差异、重复计算稳定性和三个类别 argmax。seed 42 的第一版严格 `1e-6` smoke 门槛发现约 `4.95e-6` 的保存前/重载后 CPU 浮点差异；两个独立重载及各自重复计算完全一致，三个 argmax 均不变，最小前二 logit 间距约 0.0179。该第一版包保留为**拒收诊断产物**，不会参与推理。最终脚本使用纯绝对容差 `2e-5`、重复计算容差 `1e-7`，并仍严格检查 argmax；三个正式包均从头训练，以保证元数据的脚本哈希与实际训练源码一致。

## 正式包与核验

正式产物均在服务器项目目录的 `outputs/`，并已完整镜像到本地项目的 Git 忽略 `outputs/` 下；下表目录名在两端一致。脚本 SHA-256 是 `2653db83407da1531598f80e0d02b4d5c958cc086048a1a540b485146aa112e8`，三包元数据的训练源码字段均与该值一致。每个包有 8 个文件。

| seed／目录 | 包总字节数 | 权重 SHA-256 | 元数据 SHA-256 | 保存前／重载后最大 logit 绝对差 |
| --- | ---: | --- | --- | ---: |
| 42：`e16_modernbert_full_seed42_v2` | 599,783,789 | `86969ca4297cf36361c6fe2371c39349ff217840243305abd416076694a3f357` | `e0482f3080e135b7d7118ba63ed1ff5412602f13af3c4749ceaaf969d2e922fe` | `5.9605e-6` |
| 17：`e16_modernbert_full_seed17_v2` | 599,783,778 | `d26ee2489493cdf40f56c6278e35863e7164ee4fc6bc6accb530cd59bc07d44a` | `048a55a1def13223365ba41696feba8c32da4b4c20892ddd906cb0592920be77` | `6.3479e-6` |
| 73：`e16_modernbert_full_seed73_v2` | 599,783,760 | `789cbe5b0cbfef3e3c6bcc63f2b4a9afba76d991aebfcfdb9dc18cb56a7fb93e` | `e1e2c284e8135b355b333ba7b509649cb724784a0697aac322352b8cc9c2fe22` | `4.6492e-6` |

三包各以 `TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2` 离线训练，固定最后第 12 轮，不进行 checkpoint 选择。训练耗时分别约 249/257/261 秒；末轮训练损失分别约 0.000303/0.000257/0.000572，跳过的 fp16 优化器步数分别为 5/3/7。这些仅作为训练过程审计记录，不代表验证准确率。

训练进程的保存后测试、独立新进程的 `verify --device cpu` 均对三个包成功。保存前内存模型重复计算、重载模型重复计算的最大差在三个包里均为 **0**；每包三个固定文本的 argmax 在保存前后均一致。烟雾检查的最小 top-2 margin 分别是 0.308、0.421、0.786，均远高于观察到的浮点差。三个本地镜像再次按 `metadata.artifact_sha256` 逐文件读取、计算 SHA-256，与服务器包的权重及元数据哈希一致；没有只复制部分权重或仅凭文件大小判断完成。

复现单个包的命令模板（`SEED` 与 `OUTPUT` 分别用上表对应值，输出必须是新目录）。运行前将 `PRETRAINED_SNAPSHOT` 设为本地 `answerdotai/ModernBERT-base` revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` 的快照目录：

```bash
TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 python fit_modernbert_full.py fit \
  --seed SEED \
  --pretrained-snapshot "$PRETRAINED_SNAPSHOT" \
  --output-dir outputs/OUTPUT
```

独立重载命令：`python fit_modernbert_full.py verify --package outputs/OUTPUT --device cpu`。

## 外部验证限制

三个完整模型的本地烟雾检查只验证保存、加载和少量预测一致性；它不测泛化。Kaggle 提交前还需完整断网推理模拟、约 4,000 行测试规模的时间与内存检查，并确保与已经发布的 E0c、RoBERTa 完整包融合时标签顺序、文本预处理、分数标准化与 E10/E16 OOF 完全一致。
