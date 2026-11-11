# E30-A 全量训练与离线部署准备

**状态：仅有可审阅代码；未启动全量 GPU 训练，未生成部署权重，也未提交 Kaggle。** E30-A 五折对原 E24 固定融合有提升，但截至本计划，按保守的“已存在固定候选都纳入比较”口径，E24+E30 相对 E24+E28 的某一类召回下降超过预登记的 0.05 上限。晋级裁定未通过时，下面的 `fit` 和 `verify-reload` 命令必须保持停用；脚本本身也会重算该比较并拒绝运行。

## 独立全量脚本

[`fit_e30_full_official.py`](../fit_e30_full_official.py)只复用锁定的 [`run_e30_supervised_embedding8b.py`](../run_e30_supervised_embedding8b.py)的模型构造、tokenizer、数据加载器、AdamW 训练、adapter 保存与重载函数；不改 E30 v3 配置或五折验证代码。全量拟合在候选晋级后才使用官方 `train.csv` 的 **2,699 行 / 150 prompt**，不加入 E24 外部数据。保持 seed 42、最后非 padding token 读出、256 token、batch 1、梯度累积 8、3 epoch、普通交叉熵、LoRA r16/alpha32 及唯一一个部署 adapter。全量数据使每轮步数和 warmup 步数增加，这是同一固定训练规则应用到更多样本的自然结果。

脚本提供三个独立阶段：

1. `check`：CPU 核对官方训练 SHA、E30 v3 配置和协议锁、五折 `run.json` 和 OOF 文件哈希、资源报告、逐折/逐类聚合、E36 固定比较器报告、独立审计报告及晋级记录，再逐文件核对 8B 官方模型快照 SHA。它不加载模型到 GPU，也不拟合。
2. `fit`：重复全部检查，确认 3090 空闲和包版本后，训练唯一的官方全量 adapter 与 18 类线性头，保存 tokenizer、3 条固定无标签 smoke logits、每个工件哈希、资源及来源元数据。权重只写入 Git 忽略的 `outputs/e30_full_official_v1/`；没有保留集或测试集准确率计算。
3. `verify-reload`：在新的 GPU 进程中再次核对全部文件，重载 adapter 与分类头，比较 3 条 smoke 的 18 维 logits 和 argmax，再对官方训练文本循环构成的 **4,000 条无标签**固定代理进行完整推理，记录时间、显存、有限值和格式。它不输出逐行分数、回答文本或预测标签。

`check`、`fit`、`verify-reload` 都要求审计后才会生成的 `outputs/e30_fullfit_promotion_record.json`。此记录是实验晋级状态及已审计文件哈希的机器凭据，**不是用户重复授权请求**。当前不应创建“晋级”记录；即使错误创建，脚本仍会从 E36 聚合表重算 E24+E30 对 E24、E24+E28 中最高分旧候选的三项门槛，并在本次结果上拒绝全量训练。若将来研究方案改变比较口径，须另开协议版本，不能改写本脚本以绕过原实验门槛。

晋级记录的固定字段是：`decision=e30_a_promoted_against_strongest_prior_fixed_candidate`、`approved_fusion=e24_plus_e30_fixed_weight_1`、`comparison_policy=strongest_prior_fixed_including_e28`、`strongest_prior_fixed_candidate`，以及 `official_train_sha256`、`e30_script_sha256`、`e30_config_sha256`、`e30_protocol_lock_sha256`、`e30_resource_report_sha256`、`e30_analysis_metrics_sha256`、`e30_fold_metrics_sha256`、`e30_class_metrics_sha256`、`e30_analysis_oof_sha256`、`e28_analysis_oof_sha256`、`e36_analysis_metrics_sha256`、`independent_audit_sha256`。所有 SHA-256 必须来自服务器上实际训练/审计文件；不能用本地换行转换后的文档哈希代替。

如未来另一个经审计的 E30 结果确实晋级，在服务器项目根目录、资源空闲且晋级记录存在后，三个阶段的命令依次为：

```bash
python fit_e30_full_official.py check \
  --model-dir data/models/qwen3-embedding-8b-1d8ad4c \
  --audit-report docs/e30_independent_audit.md \
  --promotion-record outputs/e30_fullfit_promotion_record.json

CUDA_VISIBLE_DEVICES=<idle-3090-GPU-UUID> python fit_e30_full_official.py fit \
  --model-dir data/models/qwen3-embedding-8b-1d8ad4c \
  --audit-report docs/e30_independent_audit.md \
  --promotion-record outputs/e30_fullfit_promotion_record.json

CUDA_VISIBLE_DEVICES=<idle-3090-GPU-UUID> python fit_e30_full_official.py verify-reload \
  --model-dir data/models/qwen3-embedding-8b-1d8ad4c \
  --audit-report docs/e30_independent_audit.md \
  --promotion-record outputs/e30_fullfit_promotion_record.json
```

`--audit-report` 的实际文件名要以独立审计产物为准；上述路径仅说明接口。现有 E30 资源试验在 1,728 行上一轮约 18.3 分钟，完整 2,699 行三轮训练的耗时需实际测量，不能直接把测试速度当成 Kaggle GPU 速度。

## 最小固定 E24+E30 推理与 Kaggle 包

若后续有通过门槛的版本，部署分数必须完全复现预登记公式：

`row-z(E24 E0c) + row-z(RoBERTa) + mean(row-z(ModernBERT seed 42/17/73)) + row-z(E30 final logits)`。

类别顺序沿用 `run_e0.LABELS`，分支系数固定为 1.0，不在测试集或公榜上调权重。E24 三路可先复用已验证的 CPU 推理和三个现有私有模型数据集；E30 只新增一个全量 adapter + 线性头 + tokenizer + 元数据的数据集，并挂载与训练时 SHA 一致的官方 8B embedding 底座。若 Kaggle 官方模型资产的具体文件哈希与已锁 revision 不一致，需要另建私有底座数据集；不能静默换底座。模型资产按 Apache-2.0 独立分发，MIT 代码仓库不含权重、训练/测试 CSV 或逐行 OOF。

Kaggle 侧还需要可离线安装或已预装且核对版本的 **GPU** PyTorch 2.6.0、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0、bitsandbytes 0.45.5 及对应 CUDA 依赖。现有 E24 wheelhouse 是 **CPU** PyTorch，不能用于 4-bit E30 推理。要在与隐藏重跑相同的 Kaggle GPU 类型上，先验证本地文件加载、4,000 条无标签完整推理、总时长与显存，以及 `submission.parquet` 的 `id, model` 顺序、行数和 18 类合法性；服务器 RTX 3090 资源通过不能证明 Kaggle GPU 等效。该验证只有在新的全量权重存在后才有意义；目前无上传或提交动作。
