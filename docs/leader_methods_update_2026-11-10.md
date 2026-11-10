# Name That LLM：公开 0.680 方案复核（2026-11-10）

各实验与后续路线的最新总览见[实验总览](experiment_catalog_and_roadmap_2026-11-10.md)。

## 可核实的结果与边界

[V63 公开 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484)显示公榜 balanced accuracy **0.680**；我们的 E24 隐藏重跑已成功，公榜为 **0.434**，两者相差 **0.246**。E24 的 **0.425** 是开发集按 prompt 分组的折外分数，不能拿它与公榜直接比较。我们的 E16 公榜为 **0.407**。比赛尚未结束，没有最终私榜名次。

该 Notebook 的可见运行时间为 **21 秒**，代码先用三条公开样例的 ID、字节长度及文本指纹检测固定预览，命中时直接写出格式样例，并注明没有加载权重、没有验证隐藏测试耗时。其余输入走完整集成推理。因此 21 秒不代表约 4,000 条隐藏测试的运行时间或训练耗时。[V63 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484)

## 相对上一版新增的机制

作者说明在原有 0.666 多模型、稀疏文本与排版集成的基础上，加入 **Qwen3-Embedding-8B 的五个 prompt 折监督 LoRA 分类器**。该分支读取 36 层中的八个深度，使用学习到的归一化、原文与按 prompt 中心化的两个分类头；长文本使用覆盖式 512-token 窗口。它在几何混合中的标称占比为 **40%**。原父集成的 Qwen3.8-27B 分类器只占其余分支内部 **3%**。训练权重、原始折校准和设计归功于 [CHEN Han 的 0.679 Notebook](https://www.kaggle.com/code/concyclics/0-679?scriptVersionId=365653660)；V63 公开了推理代码和工件引用，但这些材料没有单组件消融，不能把 **0.680−0.666** 全部归因于新分支。[V63 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484) · [0.679 Notebook](https://www.kaggle.com/code/concyclics/0-679?scriptVersionId=365653660)

这里的 **Qwen3-Embedding-8B** 是专门的 embedding 骨干经监督适配；我们的 E28 则是锁定 `Qwen/Qwen3-8B` 生成基座的十八分类 QLoRA 资源试验。二者不能视为同一个模型或相互替代。我们之前测试过冻结 **Qwen3-Embedding-4B**，但不能由其结果判断 8B 骨干经 LoRA 监督训练后的效果。[E28 记录](e28_qwen8b_lora_feasibility.md) · [冻结 embedding 对照](e1_frozen_embeddings.md)

## 对下一轮的具体启发

1. **数据优先。** 高分方案依赖多来源外部训练文本及预拟合工件；我们的 E24 小外部源在开发集和 Kaggle 公榜均有增益，来源生成协议依赖完整官方训练统计的问题仍需独立检验。后续外部源逐行核对模型路由、生成协议、提示选择、许可与近重复，并做单源增量消融。[E24 来源审计](e24_prvsiyan_source_audit.md) · [E26 分布诊断](e26_domain_audit.md)
2. **单独立项做监督 embedding LoRA。** E30 已固定可核验的 `Qwen3-Embedding-8B` revision、完成 CPU 协议预检并开始下载权重。接下来先测单折资源，再用简单分类头做完整五折，评估单路召回、与 E24 的互补性；有信号才试多深度头。不要把公开 Notebook 的复杂头部、融合权重或训练工件直接当作我们的已验证收益。[E30 设计](e30_supervised_embedding8b_design.md)
3. **多视角和集成放在后面。** 四视角稀疏、排版、家族门控与边际校正都有公开先例，但我们的 E20、E21、E22、E23、E26 已做过相近小规模消融，增量均未过预设晋级门槛。只有新的可信数据或独立强模型出现，才重新测它们的条件互补性。[历史复核](leader_methods_research_2026-11-06.md) · [实验结果](results.md)

所有候选仍使用相同的 120 个开发 prompt 的五折 GroupKFold；先测单折资源和单路价值，只有通过预设晋级规则才扩展五折、全量训练与 Kaggle 提交。公榜只用于少量外部核验，避免围绕公榜反复调参数。
