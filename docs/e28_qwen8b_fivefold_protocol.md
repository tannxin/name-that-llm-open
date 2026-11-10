# E28 Qwen3-8B 官方数据五折性能实验（预登记）

**状态：已在任何 E28 带标签验证之前锁定配置并完成本地 CPU 预检；尚无 E28 验证分数。** 资源试验已通过，详见 [E28 资源记录](e28_qwen8b_lora_feasibility.md)。本实验只评价官方数据上的 18 类分类效果，不引入外部样本。

## 固定输入与训练

- 基座：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`；运行前逐文件核验固定 SHA-256。它是生成模型分类头分支，不是 Qwen3-Embedding-8B。
- 训练文件：官方 `train.csv`，SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`。沿用 E0 固定的 120 prompt、2,160 行开发分区及 5 个 prompt 分组折；30 prompt 留出集不参与训练、验证、选模或融合。
- 每折独立从同一基座初始化，只用其他 96 prompt 的 1,728 行训练；本折 24 prompt 的 432 行只在最后一个 epoch 后做一次验证。训练与验证 prompt 交集必须为空；逐折共覆盖开发集每一行恰好一次。
- 原始回答文本，不加指令或聊天模板；最多 256 tokens，右 padding，微批 1、累积 8，seed 42。18 类标签顺序固定 `run_e0.LABELS`。
- NF4 4-bit、double quant、FP16 compute；LoRA `r=16, alpha=32, dropout=0.05`，目标 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，18 类线性分类头可训练。普通交叉熵、AdamW `2e-4`、weight decay `0.01`、总优化器更新的前 3% linear warmup 后 linear decay；固定 3 epoch，不依验证结果选 epoch、阈值或权重。
- 单折保存 adapter、tokenizer、验证行的 18 类 logits、`id/prompt_id/true_model/fold/seed`、运行元数据。`fit-fold` 前必须检查通过的 E28 资源报告、完整模型 SHA、原始数据与代码协议锁，以及物理 GPU2 空闲和 CUDA UUID。

机器配置是 [`configs/e28_qwen3_8b_fivefold.json`](../configs/e28_qwen3_8b_fivefold.json)，执行代码是 [`run_e28_qwen8b_fivefold.py`](../run_e28_qwen8b_fivefold.py)。`preflight` 在 CPU 上验证官方样本维度、prompt 隔离、五折覆盖，并将训练代码、配置、资源测试脚本、E0 和 E24 比较来源的 SHA-256 写入新的 `outputs/e28_qwen8b_fivefold/protocol_lock.json`。当前本地预检输出：2,160 开发行、120 prompt、`validation_accuracy_computed=false`；锁 SHA-256 `ed4577e6d61215850d67e53ce0a069b884da3a0d411073bc3f1c28a881636bf9`。服务器启动前会独立运行同一预检；若代码或输入变更，必须另开协议版本，不能修改锁后继续评分。

## 预先固定的比较与晋级

仅在五折全部成功并对齐原始行顺序后汇总；不报告单折为最终模型效果。先逐行精确复现 E24 固定融合：`row-z(E24 E0c) + row-z(RoBERTa) + mean(row-z(ModernBERT seed 42/17/73))`，其已归档 OOF balanced accuracy 必须为 `0.425`。然后**固定 1.0 权重**加入 `row-z(E28 logits)`。不扫描融合权重或用验证标签挑选后处理。汇总单路 E28 BA、融合 BA、逐折/逐类召回、混淆矩阵、正确/错误互换数和按 120 个 prompt 配对的 20,000 次 bootstrap 区间。

相对 E24 的候选晋级门槛同时满足：融合 BA 至少提高 **0.01**，至少 **4/5 折不下降**，任一类召回下降幅度 **小于 0.05**。若汇总时已有更好的固定参照，还须与新参照比较。开发集已经用于多轮选模，因此通过门槛仅表示值得做完整离线部署核验，不等同 Kaggle 公榜提升；不会使用留出集进一步挑选。

## 执行与资源协调

服务器端通过 [`launch_e28_fivefold.py`](../launch_e28_fivefold.py) 对 GPU2 串行运行 `preflight`、折 1–5、`summarize`。launcher 在每折开始前等待实际 GPU2 空闲；进程退出且显卡确实空闲后才写 `outputs/e28_qwen8b_fivefold/gpu2_released.ok`，供后续 E30 排队。E28 成功或失败记录在同目录的 `gpu2_release_status.json`；释放标记本身只表示 GPU2 已空闲。后台训练日志独立于 Git，并严格保留失败时的部分输出以供审计。完整五折预计需约 5–6 小时，估计来自单折一轮 18.19 分钟的资源实测，并非完成时间保证。

截至本节记录，已发现之前使用的 SSH Pod 端口返回 `Connection refused`；本地 CPU 预检完成，但服务器部署和 GPU 五折尚未启动。连接恢复后先核对服务器输入与 GPU2 空闲，再启动 launcher。**目前没有 E28 性能结果。**

本地还对固定 E24 对照做了 CPU 自检：从五路已归档分数逐行重算，2,160 个预测与已保存的 E24 OOF 完全相同，balanced accuracy 精确复现 `0.425`。E28 训练脚本和 launcher 均通过 Python 语法检查；GPU 训练及五折汇总仍待服务器执行。
