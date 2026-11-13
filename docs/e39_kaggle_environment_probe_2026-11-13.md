# E39 Kaggle T4 环境探针（2026-11-13）

已在私有 Kaggle Notebook `probe_e39_t4_environment_only` 以 **T4×2、Internet off** 运行无评分环境探针。没有挂载比赛数据或模型，没有生成提交文件；探针结束后已停止 GPU 会话。这次探针只检查环境，不代表 E39 可部署。

| 检查项 | 观察值 |
| --- | --- |
| GPU | 2×Tesla T4；每张总显存 15,360 MiB，运行前空闲 14,912 MiB |
| `/kaggle/working` | 可写，空闲 19.5 GiB；应从只读输入直接加载底座 |
| Python / CUDA | Python 3.13.16；Torch 2.11.0+cu130；CUDA 可用，2 张卡 |
| 实际依赖 | Transformers 5.20.2、PEFT 0.21.2、Accelerate 1.15.0、bitsandbytes 未安装 |
| 锁定训练依赖 | Torch 2.6.0、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0、bitsandbytes 0.45.5 |
| Parquet | pyarrow 23.0.1，可导入且内存回读通过 |

**结论：GPU 数量、预加载空闲显存和 Parquet 检查通过，但五个锁定 GPU 依赖均未匹配，其中 bitsandbytes 缺失。** 因而当前 Kaggle 镜像不能直接执行 E39 固定 NF4 推理。尚未测底座和 adapter 加载、实际推理显存或完整 4,000 行耗时；不能据此拟合全量 adapter 或提交比赛。

下一步先确定 Internet-off、Python 3.13 环境可用的离线 GPU 依赖工件及许可证，并做无评分导入/加载测试。若锁定版本在此镜像无法安装，可另建**只用于部署兼容性验证**的版本化方案，要求服务器和 Kaggle 对同一训练侧三文本的 18 维 raw logits 进行固定容差比较；任何分数偏差或完整负载失败都停止，不依据 OOF 或榜单调权。服务器上三文本参考已完成并通过结构核验，仍保存在 Git 忽略的私有目录，没有把文本或逐行 logits 放入仓库。
