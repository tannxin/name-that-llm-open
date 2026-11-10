# Name That LLM：下一轮可执行优化方案（2026-11-06）

## 基线与判断

官方指标是 18 类平均召回率。现有 120 个开发 prompt、2,160 行的五折 `GroupKFold` 结果：E10 为 **0.399537**，其 Kaggle 公榜为 **0.401**；E16 三种子 ModernBERT 固定融合为 **0.409259 OOF**，比 E10 多判对 21 行，但按 prompt 配对的 95% 区间 **[-0.004630, +0.024074]** 跨零。**E16 已于 2026-11-06 12:37 北京时间提交，隐藏重跑成功，Kaggle 公榜 0.407**；与 E10 同一公榜指标相比高 0.006，不把 OOF 与公榜分数相减。30 个保留 prompt 已评估一次，随后进入 E16 全量训练，不能再称为未见保留集。[结果](results.md) · [E16 复核](e16_modernbert_seeds.md)

E10 的错误约 59% 是家族正确而版本错误；`gemini-3.7-flash → 3.6-flash` 有 51/120，`gpt-5.6-sol/terra → luna` 各有 39/120。E16 的家族内版本条件正确率由 0.5285 升到 0.5443，但 `gpt-5.6-luna` 召回下降 7/120。同题字符距离与双向混淆相关（Spearman −0.740），不过易混 Gemini 版本对的跨 prompt 差异方向一致性约 0.095，接近随机参考 0.091；相似度不能直接当成加大损失的依据。[误差](e10_error_analysis.md) · [距离诊断](pair_geometry_diagnostic.md)

## 顺序、对照与停止条件

以下阈值是**下一轮预先建议的实用门槛**，并非统计显著性或已有实验结论。每项先锁定来源版本、模型 revision、随机种子、融合规则和停止条件，再运行。

| 顺序 | 唯一主要对照与产物 | Go / No-go | 资源上限或核算 |
| --- | --- | --- | --- |
| 0. E16 外部检验 | **已完成：**2026-11-06 12:37 北京时间提交既定断网 Notebook；Kaggle hidden rerun 成功，公榜 **0.407**，上一版 E10 **0.401**。公榜只检查迁移，不用于删 seed 或改权重。 | 已完成成功提交并记录分数；私榜尚未知。 | 本地 4,000 行 CPU 端到端推理 **83 分 24 秒**。[交接](e16_kaggle_handoff.md) |
| 1. 审计外部数据 | **已完成：**prvsiyan v9 原始 `new` 5,400 行逐条与 API envelope 对照；严格路由白名单保留 **4,322 行、17 类**。Qwen 快照不明、Kaggle proxy 后端不明，均排除；Open Pool v15 因用全官方标签选源而不进入五折。 | 行级路由与返回模型核验通过；来源协议曾参考全官方训练长度统计，后续 OOF 只作探索性。 | [来源审计](e24_prvsiyan_source_audit.md) |
| 2. 低成本数据消融 | **已完成：**从合格来源固定哈希取 250 个外部 prompt、3,620 行，每类外部／官方总训练权重 **0.25**，同五折重拟合 E0c；只替换 E16 中旧 E0c。 | 固定融合 **0.409259→0.425000**，+0.015741；**4/5 折**不降，最差类 −0.041667，预定三项门槛均通过。 | 已完成五次原环境 CPU 拟合；[E24 报告](e24_prvsiyan_external_ablation.md) |
| 3. 两阶段监督训练 | 首选同一 ModernBERT-base revision：A 仅官方训练的已有 seed 42；B 官方＋审计合格外部行混合训练；C 外部训练后只用官方折训练行适配。固定 18 类分数逐行标准化，分别比较单路、替换原 ModernBERT 一路后的三路融合，并对照 E16。 | 候选在五折与两个种子上方向一致，较 E16 的固定融合实用增益 **≥0.01**、至少 4/5 折不降、无单类下降 ≥0.05，才全量拟合；否则保留诊断，不继续扫权重。 | 先 **1 折只测显存和耗时**，不据其分数选方案；可行后 B/C 各 5 折、首轮 1 seed，达门槛再补第二 seed。旧 ModernBERT 五折约 **802 秒／RTX 3090**；外部数据使耗时增加，须实测。 |
| 4. 更大监督模型与融合 | 仅在前述数据收益或模型互补性明确后，测试 **Qwen 8B LoRA 18 类分类**的单折可行性，再做完整五折；将其作为一条新分数路与 E16 低自由度固定融合。 | 同上，另须 4,000 行断网推理在 Kaggle 可用资源与 9 小时内完成；否则不扩大到 27B。 | 先 1 折 GPU 显存／训练时间与 4,000 行推理预算；不能用公开 Notebook 的 21 秒提交时间推算训练成本。 |

外部数据现以 [prvsiyan v9 原始来源](e24_prvsiyan_source_audit.md)为主：逐行 API 返回身份可核，固定 17 类路线已进入 E24；`qwen3.8-max` 仍缺与比赛目标快照对应的公开证据，[Will Rice v5](https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest) 的 Qwen 路由也无法消除这一缺口。来源生成时看过完整官方训练集的长度统计，因此 E24 虽过五折门槛，仍需独立提交或新来源复核。**Open Pool v15 主文件**曾使用全部 2,699 条官方回答和 150 个 prompt 选择来源/路由，仅禁用 `match_weight`、`style_score`、`max_cos_train` 不能消除这一污染；本轮不把它用于官方 prompt 五折。[旧 E12 审计](e12_external_data.md)

## 穿插进行的低成本方法消融

**E20 已执行。** 唯一预登记的 `E0c + shape` 将 E0c 单路五折 BA 从 0.342593 提高到 0.353704；替换固定 E16 融合中的 E0c 后，BA 从 0.409259 到 0.413426，增量只有 0.004167，配对 prompt bootstrap 区间跨零，低于上表预设的 +0.01 融合晋级门槛。因此保留为诊断，不替换当前 Kaggle 主方案，也不为它新增公榜提交。[E20 结果](e20_shape_ablation.md)

公开 [0.666 Notebook V47 源码](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) 提供三个可在现有分数和 CPU 上先验证的假设。它的 0.666 是多作者、多模型与外部数据的合成成绩；下列每项须独立改变一个因素，并在相同官方 prompt 分组折上比较，**不直接套用其 12、85/15、0.30 等权重**。[源码核查](leader_methods_research_2026-11-06.md)

1. **已完成：一个固定参数的 shape 消融。** E0c 已含词 1–2 gram、原文字符 3–5 gram 和 57 个风格特征；[E8](e8_fingerprint_ablation.md) 又试过精确标点率、功能词 1–3 gram 的特征拼接，单路与融合增益均小且 prompt bootstrap 区间跨零。本轮只在原 E0c 加入 **一块** shape TF-IDF：把连续 Unicode 字母替换成 `X`（整段字母均为大写）或 `x`（否则），保留数字、标点、空格；固定字符 2–6 gram、`lowercase=False`、`sublinear_tf=True`、`min_df=2`、`max_features=60000`、块权重 0.5，其余 E0c 分类器和训练程序不变。已重拟合官方训练侧五折，比较 E0c 单路及替换 E16 内 E0c 分数后的固定融合；不扫 n-gram、权重或新的标点／功能词方案。后续数据来源实验单独开展。公开四视角 SVC 与约 19 万条 Open Pool 同时出现，没有 shape 单项消融，不能认定它必然有效。
2. **已完成：守恒家族概率的软门控。** 官方训练折内的兄弟模型专家按固定强度重排，五折 E16 **0.409259→0.410185**，仅 +0.000926；3/5 折不降，最差类 −0.075。未达预设晋级门槛，不调整强度或挑选家族，也不替换主方案。[E21 结果](e21_soft_family_gate.md)
3. **排版证据、类别边际校正均已单独检验。** [E22](e22_layout_evidence.md) 在低置信文本上加入折内排版出现证据，E16 **0.409259→0.410185**，仅 +0.000926；官方训练 CSV 不含实际换行或反引号，行级 Markdown 线索覆盖不足。[E23](e23_marginal_correction.md) 以每折无标签预测分布作一次有界类别边际校正，E16 **0.409259→0.412963**，仅 +0.003704；两者均低于晋级门槛，不组合、不提交。隐藏测试类别比例未知，不能强制每类等数或用测试聚类匈牙利分配。

这三项的晋级判据沿用上表第 2 步：记录逐折／逐类变化、120 prompt 配对 bootstrap，并要求固定融合出现有用且稳定的增益。每次只预登记少数独立候选，报告同一开发集上的多重筛选风险。公开方案使用的 `match_weight` 在旧 [E12 审计](e12_external_data.md) 中依赖全部 150 个官方 prompt；若用它按相似度筛选外部数据，会污染我们的外层验证折。来源选择须只看原始来源元数据，或在每个训练折内重新拟合。

若第 2 步的便宜模型无收益，但来源审计显示足够的精确版本覆盖，可预先登记**一次**第 3 步单折表征试验；其分数不作为调整参数的依据。冻结骨干→只解冻顶部层可作为独立训练配方消融，须与现有全量微调同折同数据比较，不能与 B/C 同时改变。所有晋级还需报告 120 prompt 配对 bootstrap 区间，明确同一开发集多轮试验的选择偏差；区间下界大于零会增强证据，但单个小幅正增益不自动证明泛化。

## 从公开方案借鉴什么

| 原始来源 | 转化为本赛实验；不能照搬的部分 |
| --- | --- |
| SemEval-2024 Task 8 B 冠军 [AISPACE 论文](https://aclanthology.org/2024.semeval-1.212.pdf)、[官方报告](https://aclanthology.org/2024.semeval-1.279.pdf) | 同标签数据去重与扩充先于复杂模型，之后按逐类互补性融合；其六分类准确率不能与本赛 18 类平均召回率比较。 |
| SemEval 亚军 [UniBuc 论文](https://aclanthology.org/2024.semeval-1.63.pdf)、[代码](https://github.com/ClaudiuCreanga/semeval-2024-task-8) | 保留原文标点；在固定折上单独比较冻结骨干→解冻顶部层与现有全量微调，而非把清洗或冻结当默认改进。 |
| Kaggle DAIGT 冠军 [复盘](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/comprehensive-1st-place-write-up)、[代码](https://github.com/rbiswasfc/llm-detect-ai) | 借鉴多生成器／提示／采样协议覆盖和互补表示；其二分类 AUC 排名融合不直接适合本赛 18 类指标。 |
| Kaggle DAIGT 亚军 [复盘](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/guanshuo-xu-2nd-place-solution-with-code-and-data)、[推理](https://www.kaggle.com/code/wowfattie/daigt-2nd-place) | 借鉴外部训练→目标域适配及防范 CV／公榜分布差；外部人类／机器标签不能替代 18 个精确模型版本。 |
| 本赛公开 [0.666 V47 Notebook](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) | 借鉴经审计外部数据上的四视角 SVC、守恒家族总概率的软门控、低置信排版证据和低自由度互补融合；其 Qwen3.8-27B LoRA 只占最终概率 5%，分数不能单独归因给它。 |

该公开 Notebook 的 21 秒可见运行仅对精确匹配的三行公开样本写预览结果，完整模型前向被跳过，不能用作隐藏约 4,000 行的推理预算。它复用 Roman Tamrazov、mercurydust、Areeb Ali 及其他作者的模型或拟合状态；Notebook 页面未给出足以拆分 0.666 的独立消融。我们已有冻结 Qwen 4B 为 0.2005、E4 同题 SupCon 低于交叉熵、E17 家族头为 0.3986、E18 成对 margin 为 0.4023、E19 句窗裁剪为 0.4037，因此暂停继续枚举冻结 embedding、硬家族路由和复杂损失。[全表](results.md) · [公开方案核查](leader_methods_research_2026-11-06.md)

## 泄漏与部署边界

- 五折仍以官方 `prompt_id` 分组。TF-IDF 词表、缩放器、外部数据筛选规则中任何依赖官方样本的统计、模型、校准器和类别对选择，均只在当前外层训练的 96 个 prompt 内拟合；需拟合融合权重时再做内层 prompt OOF。外层 24 个验证 prompt 只用于一次评价。
- [Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) 的 212k 旧审计池缺逐行真实模型／路由；其 `match_weight`、`style_score` 等字段使用全部 150 个官方 prompt，**不得**用于五折外部样本筛选或加权。它仍可进入上文定义的弱标签探索路线，但须剔除这些列、记录误标与提示选择偏差，并做来源剔除敏感性检查。[E12 审计](e12_external_data.md) 先前 [Will Rice v5](e3_public_data_access.md) 仅八类约 2,000 行的低权重试验未胜过 E0c，不能推断可靠的 18 类扩充必然无效。
- 30 个保留 prompt 已使用，并已进入全量训练；Kaggle 公榜也不参与反复选参。外部 prompt 与官方 prompt 的重合若缺原始提示而无法核实，应标记为未知并做来源剔除敏感性检查，不把它写成“已无泄漏”。测试文件只有 `id,text`，不使用需要测试 `prompt_id` 的硬匹配或匈牙利分配；已有[公开反例](https://www.kaggle.com/competitions/name-that-llm/discussion/751802)显示该规则的 OOF 提高可伴随公榜下降。
- 本赛[提交要求](https://www.kaggle.com/competitions/name-that-llm/overview/description)允许**公开且免费可获得**的外部数据及预训练模型；数据审计也要核实这一可获得性，不能仅凭本地 API 日志视作合格来源。
