# E39 Kaggle 离线部署验证计划

E39 的五折结果已通过预登记的 E24 相对离线门槛，Kaggle 部署仍需实测。本计划只验证固定 E39 分支能否与现有 E24 分支在比赛硬件上完成推理；测试使用已保存的折 1 adapter，不训练全量模型、不读取保留集标签、不用公榜调参。

## 已知约束与风险

- [比赛说明](https://www.kaggle.com/competitions/name-that-llm/overview/description)要求 Notebook 提交、Internet off、GPU 运行不超过 9 小时，输出 `submission.parquet`，含唯一的 `id` 和 18 类合法 `model`。交互版 `test.csv` 只有少量样例，隐藏重跑约 4,000 行；样例成功不代表完整运行成功。
- Kaggle [T4×2 公告](https://www.kaggle.com/product-announcements/735239)给出每张 T4 16 GB。服务器 RTX 3090 的 E39 资源探针用 4,000 条无标签训练侧文本推理耗时 629.97 秒，**峰值预留 15,885,926,400 B（14.80 GiB）**。这接近一张 T4 的容量，单卡加载可能 OOM；必须在实际 T4 上核对，不能按 3090 外推。
- E16 既有 CPU 全流程在 4 线程服务器上对 4,000 行耗时 83 分 24.582 秒；E24 额外 E0c 推理开销较小。GPU Notebook 的 CPU 时间、环境安装及 E39 T4 推理时间尚未测量。比赛的 9 小时限额以整个 Notebook 为准。
- Kaggle [Dataset 技术规格](https://www.kaggle.com/docs/datasets)当前列出每个 Dataset 200 GB、私有 Dataset 总量 200 GB；固定底座大小在额度内。模型文件应直接从只读输入挂载加载，避免将 16 GB 底座复制到有限的 `/kaggle/working`。

## 锁定资产

| 输入 | 体积及来源 | 校验 |
| --- | --- | --- |
| Qwen3-8B-Base | 固定 revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`，五个权重分片合计 16,381,516,776 B；完整运行目录实测 16,393,051,150 B | 训练前锁定的 12 文件 SHA-256；Kaggle 挂载后逐文件重新核对。若使用 Kaggle 现有公开模型资产，必须先证明文件哈希与锁定版本一致，否则单独上传固定快照。 |
| E39 折 1 adapter | 已完成三轮训练的折内工件，目录实测 190,847,477 B，其中 `adapter_model.safetensors` 为 174,950,552 B | 权重 SHA-256 为 `576ac1ff2a26c0bed24a56db90a47fc4e13f7e8596426e95cef7dd5eb8046354`，与折 1 导入清单一致；Kaggle 上还要核对其余文件哈希、18 类顺序、训练脚本和协议锁。此处仅用于无标签部署测试，不用作最终提交模型。 |
| E24 固定分支 | 已有 E0c/RoBERTa、三种子 ModernBERT、E24 E0c 私有输入 | 原 Notebook 的逐文件哈希与 smoke 检查保持；固定融合为 `row_z(E24 E0c) + row_z(RoBERTa) + mean_seed row_z(ModernBERT) + row_z(E39 logits)`，E39 系数固定 1。 |
| GPU 依赖 | 需实际 Kaggle 环境支持 PyTorch 2.6.0 GPU 构建、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0、bitsandbytes 0.45.5 | 先用 Internet off 的 T4×2 Notebook 只读探测版本、CUDA、可用显存；缺项则准备并校验 GPU wheelhouse。现有 E24 wheelhouse 是 **CPU** PyTorch，不能用于 E39 4-bit 推理。 |

预训练底座来自公开 Apache-2.0 模型；若单独上传固定快照，资产说明须保留模型来源、revision、许可与文件哈希。adapter 与 E24 工件存私有 Kaggle 输入。数据、权重、逐行 OOF 不进入 Git。Notebook 不从网络下载任何资源。

## 分阶段验证

1. **环境与文件。** 在绑定比赛数据的 T4×2 Notebook 中关闭 Internet，确认两张卡的 UUID/显存、Python/CUDA/包版本和各只读输入路径；先核对所有文件哈希、类别顺序及 E24 原有 smoke。不要从交互 `test.csv` 推断隐藏集耗时。
2. **最小 E39 加载。** 只挂折 1 adapter，按五折相同的 NF4 双量化、FP16、最长 256 token、`AutoModelForSequenceClassification` 18 类头和最终非 padding token 读出离线加载。先从折 1 训练侧预定三条无标签文本，在服务器保存带文本哈希的 smoke logits，再在 Kaggle 比较 18 维 logits 容差及 argmax；记录加载峰值、剩余显存和耗时。先尝试单张 T4；若 OOM，固定两卡层分配并重新验证 smoke，不能更改权重或用验证/榜单分数选部署方式。
3. **完整无标签负载。** 从 Kaggle 官方 `train.csv` 以 `usecols=["text"]` 只读取文本，按预定顺序重复形成 4,000 行代理。用合成的唯一 ID 标识代理行。在相同 Notebook 中运行 E24 全部 CPU 分支及 E39 分支，记录各阶段和总墙钟、GPU 峰值、CPU 内存；验证每行 18 维有限分数和固定 1×融合。代理输入不能产生任何性能指标。
4. **格式与时间门槛。** 对 4,000 个唯一代理 ID 写入并回读 `submission.parquet`，要求只有 `id,model` 两列、ID 顺序完全一致、无空值/重复、标签均属官方 18 类。连同环境准备、哈希、加载及推理的完整 Notebook 墙钟须显著低于 9 小时，保留余量；只有实际 T4×2 Internet-off 全流程通过，才考虑官方全量 adapter 拟合和正式提交。

上述验证可以只在私有 Notebook 中交互运行，或使用 **Save Version → Save & Run All** 保留完整日志；[Kaggle 提交流程](https://www.kaggle.com/docs/competitions)把随后在版本 Output 页点击 **Submit** 列为单独一步。验证阶段不点击 Submit，也不通过编辑器的直接提交快捷入口，因此不会触发隐藏测试重跑或榜单反馈。

若单卡装载失败，两卡部署是单独的工程修复，应固定设备映射并重新做上述全部无标签验证。若 Kaggle 资源或离线依赖不能满足，保留失败日志并暂停 E39 部署，不根据公榜选择替代模型。
