# Name That LLM：下一批并行实验（2026-11-11）

目标是把公榜 **E24 0.434** 向公开的 [0.680 集成](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484)所代表的高分区间推进；前三名门槛会变化，不能把 0.680 当作前三保证。E28、E30-A/B、E33–E37 已审计未晋级，详见[实验总览](experiment_catalog_and_roadmap_2026-11-10.md)及[E30-B 结果](e30b_multidepth_results.md)。重复调 E30-B 融合权重或再扫相同稀疏特征不作为本批方向。

| 路线 | 新信息和单因素对照 | 并行安排与下一道门槛 |
| --- | --- | --- |
| **E38：无提示 ID 的软同题校正** | 固定 TF-IDF 语义近邻，从 E24 分数中扣除局部提示偏置。完整五折 BA **0.4250→0.4662**，稀疏模拟 **0.4449**，但 Gemini 3.6 Flash 召回 −0.075。 | **已完成，No-go**；违反预登记逐类门槛，不调系数重试。[结果](e38_soft_prompt_neighbors_results.md) |
| **E40：训练侧同题风格残差** | 只在训练折按已知提示去掉 57 维风格均值，测试无需提示 ID。固定差分加入 E24 后 BA **0.4250→0.4190**。 | **已完成，No-go**；2/5 折不降。[结果](e40_prompt_residual_style_results.md) |
| **E39：Qwen3-8B-Base LoRA** | E28 是 `Qwen3-8B`，不是 [Base 权重](https://huggingface.co/Qwen/Qwen3-8B-Base)；公开[参赛者实验](https://github.com/PiKa919/kaggle-name-that-llm)提示 Base LoRA 可能是更有力的新分数路。只换骨干，沿用官方数据、E28 类别损失/长度/LoRA 日程。 | 固定模型与[单折资源门槛](e39_qwen3_base_resource_results.md)已独立审计通过；五折首次启动在训练前被另一作业抢占 GPU2，未产生分数。恢复脚本已审查，等待连续独占时段；公开参赛者分数不作为我们模型的预期结果。 |
| **公开高分工件复现** | 检查公开 0.679–0.680 Notebook 所需权重、许可、模型输入和完整隐藏推理资源，建立可运行参考。 | **只读资源核查并行**；公开预训练工件可能用过全部官方标签，不能并入我们的五折 OOF 比较。若可运行，单独评估并清楚标注外部参考，不把原作者成绩写成我们的实验增量。 |
| **E41：DeBERTa-v3-large** | 将 E14 的 DeBERTa-v3-base 只换为 [large 版](https://huggingface.co/microsoft/deberta-v3-large)，保持官方折、256 tokens、原损失与 12 epoch；也与公开 V63 的一个骨干同类。 | 修正版[单折资源门槛](e41_deberta_v3_large_resource_results.md)已独立审核通过；首次探针脚本错误与修正均留证。性能五折脚本和守护启动器正在独立审核，尚无验证分数。 |
| **下一路大语言模型** | [Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base) 是 29.54 GB、Apache-2.0 的可获取异构骨干；Llama 8B 另需门控访问与独立许可。 | E39 资源和离线信号确认后优先评估 14B；3090 通过也必须另验 Kaggle T4×2 显存和耗时，不把模型数量当作收益证据。 |
| **独立来源数据** | [Oleg 语料](p1_3_oleg_corpus_provenance_audit.md)及 [Open Pool v16 与四个上游来源](p1_4_pool_v16_source_audit.md)已复查，逐次请求/路由或独立筛选证据仍不足，新增严格准入 0 行。 | 继续寻求可核验的原始调用与独立提示；若取得证据，先锁来源版本、行级映射、许可和固定每类权重，用[单源 E0c 筛查](p1_2_external_source_refresh_2026-11-11.md)检验，再考虑 GPU。 |

所有新 OOF 仍使用同一 120 prompt 五折，30 prompt 已用保留集不参与选模；不根据单折或 Kaggle 公榜反复调参。固定晋级门槛是相对当时强基线 BA 至少 **+0.01**、至少 **4/5 折不降**、任一类召回下降 **小于 0.05**，另报按 prompt 配对区间。只有完整离线与资源审计都通过，才拟合全量权重、准备 Kaggle 提交。权重、原始数据及逐行 OOF 不进入 Git。
