# E39 Kaggle T4 环境与离线依赖探针（2026-11-13）

在私有 Kaggle Notebook 中以 **T4×2、Internet off** 先后运行环境和依赖无评分探针。未挂载比赛数据、E39 底座或 adapter，未生成提交文件；GPU 会话已停止。这些结果验证了运行环境，尚不证明 E39 能在比赛时限内完成推理。

| 检查项 | Kaggle 原始环境 | 隔离的 E39 v5 环境 |
| --- | --- | --- |
| GPU | 2×Tesla T4；每张总显存 15,360 MiB，探针前空闲 14,912 MiB | 两张 T4 上的微型 NF4 前向计算均通过；尚未加载 E39 底座 |
| Python | 3.13.16 | 在同一 Python 下创建无 pip 的隔离虚拟环境 |
| GPU 依赖 | Torch 2.11.0+cu130、Transformers 5.20.2、PEFT 0.21.2、Accelerate 1.15.0；bitsandbytes 未安装 | Torch 2.6.0+cu118、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0、bitsandbytes 0.45.5 均离线安装并导入 |
| 离线 wheelhouse | 不适用 | 私有输入的 52 个 wheel 全部逐文件 SHA-256 核验，总量约 2.94 GiB；宿主 pip 26.2.2 通过 `--python` 安装，耗时 94.8 秒 |
| `/kaggle/working` | 探针前可写空间 19.5 GiB | 安装后剩余 13.49 GiB；底座必须从只读输入直接加载 |
| Parquet | pyarrow 23.0.1 可导入 | 微型内存写入与回读通过 |

v5 探针 Notebook 的 SHA-256 为 `8510cb39d353129012f251843b1ce164dd7ca1fa55b155ab6edc7a728d29a3c0`；私有 wheel manifest 的 SHA-256 为 `a9a373ba8cf9d688b7730bdef811d4bd578416bda66bc1200f35af6accc881b8`。v4 曾在系统 `venv` 的 `ensurepip` 步骤失败，未开始 wheel 安装；v5 改为 `venv --without-pip`，由宿主 pip 的 `--python` 功能离线安装到该隔离环境。v5 的预检、安装和微型计算均通过。

**当前结论：锁定的 E39 GPU 依赖可以在 Kaggle T4×2、Internet-off 环境离线安装并运行微型 NF4 计算。** 尚未验证固定 Qwen3-8B-Base 与折 1 adapter 的完整 SHA、加载峰值、三文本 raw logits 一致性、4,000 行无标签推理、E24 联合推理或九小时总时限。因此不据此训练全量 adapter 或提交比赛。

下一门槛是把固定模型和折 1 adapter 作为私有只读输入挂载，逐文件核验后，在相同依赖环境中进行无评分加载和训练侧三文本 raw logits 对照。若单张 T4 显存不足，先固定双卡分层加载方案，再重复一致性与无标签耗时检查；不使用 OOF、保留集或榜单分数选择部署方案。数据、权重和逐行 logits 不进入 Git。
