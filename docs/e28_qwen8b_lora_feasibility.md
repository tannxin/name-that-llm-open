# E28：Qwen3-8B 十八分类 QLoRA 可行性预登记

## 固定方案（训练或查看任何 E28 验证分数之前）

目标只验证一条新骨干能否在服务器 RTX 3090 上训练、并在 Kaggle 的离线与 9 小时限制内推理。**本轮不报告验证准确率，也不选择任何融合权重；不启动五折。** 若资源可行，须先将资源报告交给项目主任务，再决定是否开展完整五折性能实验。

基座固定为公开的 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) revision `b968826d9c46dd6066d109eabc6255188de91218`（[Hub 元数据](https://huggingface.co/api/models/Qwen/Qwen3-8B)；[Apache-2.0 许可](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/LICENSE)）。该版本约 8.2B 参数、五片 safetensors 共约 16.4 GB，`transformers >=4.51.0` 支持 Qwen3。测试只用官方 `train.csv` SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`，复用 E0/E16 的 120 prompt / 2,160 行开发分区与五折 prompt 分组索引。只训练**折 1**的另外 96 个官方 prompt、1,728 行；432 条折 1 验证文本及标签均不参与训练。此轮不用 E24 外部数据，以单独识别新骨干的显存与时间成本，并避开外部数据来源协议依赖。

使用 `AutoModelForSequenceClassification(num_labels=18)` 的最后非 padding token 线性 18 类头（[Transformers Qwen3 分类文档](https://huggingface.co/docs/transformers/v4.51.1/model_doc/qwen3)）；输入为原始回答文本，无指令、无聊天模板、无生成，动态 padding、最多 256 tokens。标签顺序固定 `run_e0.LABELS`，tokenizer 的 `pad_token_id` 写入模型配置。基座用 bitsandbytes 4-bit NF4、double quant、FP16 compute（[Transformers 量化文档](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)）。LoRA：`r=16`、`alpha=32`、`dropout=0.05`、`bias=none`，目标 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，保留分类头 `score` 可训练；`prepare_model_for_kbit_training`、gradient checkpointing、`use_cache=False`（[PEFT 量化指南](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)）。

固定候选训练日程为 seed 42、3 epoch、微批 1、梯度累积 8（有效 batch 8）、AdamW 学习率 `2e-4`、weight decay 0.01、首 3% 优化器更新 linear warmup 后 linear decay、普通 18 类交叉熵、FP16 计算，不按指标选精度或轮次。为控制本次资源测试成本，**只跑折 1 的完整第 1 epoch**，其余超参按同一固定日程执行；记录吞吐与峰值，再按 3 epoch 线性外推。该外推不是训练完成证明，后续正式五折需要重新从同一预训练 revision 开始并跑满 3 epoch。折 1 验证文本仅可用于最多 32 条**无标签**前向延时检查；不加载验证标签来算分、不保存验证 18 类分数。

推理资源测试在单折权重上对官方**折 1 训练侧**文本作固定重复，构成 4,000 行无标签、与真实测试条数同规模的输入；记录模型装载、tokenization、完整 4,000 行 batch 1 前向的墙钟及 GPU 峰值。不得用折 1 验证或隐藏测试标签，也不把这份重复输入上的速度误称为 Kaggle 原生 GPU 实测。候选继续的资源门槛：GPU2 启动前空闲；包含模型加载的训练峰值保留显存 <22 GiB、推理峰值保留显存 <22 GiB、至少 95% 计划优化器更新成功、无 OOM/NaN；1 epoch ≤60 分钟且 3 epoch 线性估计 ≤3 小时；4,000 行完整 GPU 推理 ≤3 小时，叠加既有 E16/E24 管线估计仍须给 Kaggle 9 小时留至少 2 小时余量。**这些是服务器筛选门槛，最终仍需在 Kaggle 实际可用 GPU 上做离线重载与 4,000 行计时**，因为 RTX 3090 不能代表 T4/P100 的速度。

本赛[官方规则](https://www.kaggle.com/competitions/name-that-llm/overview/description)要求 Notebook 提交、Internet `off`、GPU ≤9 小时，允许免费公开预训练模型，输出 `submission.parquet`；因此若进入后续阶段，基座快照、adapter、tokenizer 和 wheelhouse 都须作为 Notebook 的离线输入重载，运行时不能访问 Hugging Face。使用单独的 E28 依赖目录与独立输出目录，不改正在运行的 E27 环境或 GPU1，不在 GPU2 有任务时启动训练。任何下载、依赖或硬件条件未满足则记录原因并停止，不替换模型、改参数或转用别的 GPU 产生不可比结果。

## 2026-11-10 预检结果：模型下载通路 No-go

服务器原环境含 Torch 2.6.0+cu118、Transformers 4.51.3，且安装的 Qwen3 实现有 `Qwen3ForSequenceClassification` 与 `score` 线性头。起初缺少 PEFT、Accelerate、bitsandbytes；已将固定的 `peft==0.15.2`、`accelerate==1.6.0`、`bitsandbytes==0.45.5` 装入独立的 Git 忽略目录 `outputs/e28_pydeps/`，没有改 E27 所用共享 venv。76.1 MB 的 bitsandbytes wheel 下载约 4 分 37 秒，实测约 306 kB/s。依赖在 E28 脚本中可导入；对不完整模型运行 `resource` 入口，按预期在 GPU 加载前即被完整性检查拒绝。

物理 GPU2 为 RTX 3090，预检时 55 MiB、0% 利用率；GPU1 正执行 E27，未被 E28 使用。模型目标目录位于 `/path/to/namethatllm/data/models/qwen3-8b-b968826d/`，所在共享盘仍有约 **17 TB** 可用，空间不是限制。固定 revision 的 Hugging Face `config.json` HEAD 可达，返回同一 commit；但五个权重分片的 HEAD 出现 `ReadTimeout` 与自动重试。下载观察时刻 1794290804 到 1794291083（**279 秒**），模型目录均为 **1,711,190 bytes，实际落盘增长 0 B**；Xet 临时缓存仅从 628,122 增到 736,986 bytes，五个分片、权重索引和 tokenizer 主文件均未落盘。16.4 GB 基座以此通路无法在合理实验时间内取得。已停止确切下载进程 PID 1216059 并确认退出，不继续反复重试。

脚本 [run_e28_qwen8b_lora_feasibility.py](../run_e28_qwen8b_lora_feasibility.py) 实现 `preflight` 与受完整快照守卫的 `resource`；前者的机器报告存于本地与服务器的 `outputs/e28_qwen8b_preflight/preflight.json`：`model_complete=false`、官方训练 SHA 正确、锁定依赖版本齐全、`CUDA_VISIBLE_DEVICES=2` 下只看见 1 块卡。脚本通过 Python 语法检查。**没有启动 GPU 训练、没有得到单折时间/显存或 4,000 行推理指标，也没有计算验证准确率。** 未来如权重可通过稳定通路完整取得，应先重新做资源测试，再决定是否开展五折；当前不能声称 8B LoRA 比现有方案有效或 Kaggle 离线可运行。

## 2026-11-10 官方镜像恢复通路：下载中

[Qwen 官方 Qwen3 README](https://github.com/QwenLM/Qwen3) 明确建议中国大陆用户使用 [ModelScope 官方 `Qwen/Qwen3-8B`](https://modelscope.cn/models/Qwen/Qwen3-8B)。从 [Hugging Face 固定 revision 文件树](https://huggingface.co/api/models/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218?recursive=true) 与 [ModelScope 文件清单 API](https://modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo/files?Revision=master&Recursive=true) 逐项核对，五个权重分片（大小分别为 3,996,250,744、3,993,160,032、3,959,604,768、3,187,841,392、1,244,659,840 bytes）及 `tokenizer.json` 的 SHA-256 和大小完全相同。`config.json`、`generation_config.json`、`model.safetensors.index.json`、`tokenizer_config.json`、`vocab.json`、`LICENSE` 也已从两站实际下载并做字节级 SHA-256 比较，完全一致。因此该镜像可提供**同一锁定权重**，无需换模型；ModelScope 仓库中的 `.gitattributes` 等平台元数据并非运行所需文件，不纳入模型快照。

ModelScope 的分片下载支持 HTTP `Range` 和 206 续传。单片 16 MiB 样本为 0.89 MB/s，五片各取 8 MiB 的并发样本总速率为 2.36 MB/s；完整运行文件约 16.397 GB，粗估需约两小时。共享盘下载前可用空间约 17.78 TB。已用 [download_e28_qwen3_mirror.py](../download_e28_qwen3_mirror.py) 在服务器后台启动 PID 1217581，下载到原固定目录，逐文件验证 SHA-256 才改名为正式文件；`.part` 支持续传。启动后约 1 分 54 秒目录已有约 239.7 MB，tokenizer 和配置文件已校验。CPU tokenizer smoke test 可离线加载，`pad_token_id=151643`、`eos_token_id=151645`，符合已写的训练脚本假设。E28 资源脚本也新增固定 SHA-256 守卫，**完整权重落盘并全数校验前不会载入 GPU**。截至本节记录时，仍未启动 E28 GPU 训练或取得性能结果。

公开 [0.680 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484) 的 Kaggle API 元数据显示其使用官方 Kaggle 模型输入 `qwen-lm/qwen-3-embedding/Transformers/8b/1`，代码中的五折 LoRA 资产为公开输入 `lightningv08/ntl-olmo2-native24-final-inference-assets/ntl-native-q8-multidepth-fivefold-0679`，该嵌入模型分支在其整体几何混合中占 40%。Notebook 另接入 `qwen-lm/qwen-3/Transformers/8b-base/1`，它是 **Qwen3-8B-Base**；E28 固定的是 **Qwen3-8B** 生成基座分类器，三者不可混为同一个 checkpoint，也不能从整体 0.680 分数归因单独模型的提升。Kaggle 的公开模型输入说明离线挂载有可参考的做法，但本 E28 锁定版本仍须在赛方 Notebook 中独立核对并完成离线重载与计时。

### 后台下载交接

最后一次只读检查：PID `1217581` 仍为 `Sl` 运行状态，已运行约 7 分 20 秒；目标目录 `1,032,392,326` bytes，起始约 `1,711,190` bytes，平均新增约 2.34 MB/s，日志未见重试或错误。检查命令（在 Windows PowerShell 执行；将 SSH 别名和 `/path/to/namethatllm` 换成本机配置）：

```powershell
$serverAlias = 'your-configured-server-alias'
ssh $serverAlias 'ps -p 1217581 -o pid=,etime=,stat=,cmd=; du -sb /path/to/namethatllm/data/models/qwen3-8b-b968826d; tail -n 12 /path/to/namethatllm/outputs/e28_mirror_download/download.log'
```

完成信号为日志出现 `COMPLETE`，且 `data/models/qwen3-8b-b968826d/e28_mirror_manifest.json` 出现 `all_sha256_verified: true`（13 个运行文件）。只有此后再次检查 GPU2 空闲、运行 E28 资源脚本的 SHA-256 守卫通过，才可启动折 1 的单 epoch 资源试验；目前没有启动 GPU 作业。

### 一次性资源测试接力（已设置，等待下载）

[watch_e28_qwen3_resource.py](../watch_e28_qwen3_resource.py) 已作为后台 PID `1219155` 启动，通用脚本以 `--download-pid` 指定下载进程。它每 60 秒只读检查上述 `COMPLETE` 和 manifest；若下载进程提前退出或等待超过 4 小时，则写入 `WATCHER_ABORT` 并退出。完成后逐项核验固定 revision、13 个运行文件的 manifest，并调用 E28 `verify_model_dir` **从本地文件重新计算全部必需文件 SHA-256**。随后再次确认物理 GPU2 显存 ≤256 MiB、利用率 0、无 CUDA compute PID；任一检查失败即退出，不换 GPU、不改参数。资源入口本身还会在加载 CUDA 模型前再次核验文件哈希与可用显存。

所有门槛均通过时，watcher 仅启动预登记折 1 的 1 epoch 和 4,000 行无标签推理，按物理 GPU2 的 UUID 设置 `CUDA_VISIBLE_DEVICES` 与离线环境变量；不计算验证准确率、不启动五折或 E30。资源入口在完整文件哈希检查之后、CUDA 模型加载之前，再核对这个 UUID 与物理 GPU2 的空闲状态，减少检查和启动之间的竞态。watcher 状态记录在 `outputs/e28_mirror_download/watcher.log`，资源子进程输出在同目录 `resource.log`，机器结果在 `outputs/e28_qwen8b_resource_mirror/resource_report.json`。最初部署检查时，watcher PID `1219155` 的日志仅有 `WATCHER_START`；下载 PID `1217581` 仍运行，GPU2 55 MiB、0%。**此时资源作业尚未启动。**

首次资源运行前的静态审查将显存门槛改为含模型加载的 `max_memory_reserved`，推理峰值也从 adapter 重载前开始测量，并新增至少 95% 优化器更新成功的判定。watcher 使用进程持有的 Linux 文件锁，退出时自动释放，防止重启竞态；若资源报告 `gate_passed=false`，它以失败状态退出，不产生成功信号。原始训练与推理超参、数据和折号均未改动。更新后的 watcher 与资源脚本须先同步到服务器，再允许启动子进程。

更新后的两个脚本已在服务器通过语法检查、与本地 SHA-256 一致；旧 watcher 已单独停止，下载进程保持运行。新 watcher PID `1220059` 在 2026-11-10 08:17 UTC 启动，日志显示 `WATCHER_START`，资源日志尚未出现。此时模型目录约 11.78 GB，仍在下载，未启动 GPU 作业。
