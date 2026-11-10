# Name That LLM：下一轮实验

这份文档记录上一轮的预定实验，其中多数已经执行。当前安排见 [第三轮优化路线](round3_plan.md)。

## 当前判断

固定的 E0c（词/字符 TF-IDF＋57 项风格特征）与修正调度器的 RoBERTa 等权排名融合，在 120 个开发 prompt、2,160 条回答的五折 OOF 上取得 **0.3833 balanced accuracy**；方案锁定后，在另 30 个 prompt、539 条回答上一次性取得 **0.4005**。这说明融合优于现有单模型，但 0.4005 不是 Kaggle 隐藏测试分数，也不应用来挑选下一轮方法；完整记录见 [results.md](results.md)。

官方指标是 18 类召回率的平均值。隐藏测试使用新 prompt，且 `test.csv` 只有 `id,text`，没有 `prompt_id`；提交须由断网、九小时内的 Notebook 生成 `submission.parquet`。[官方数据说明](https://www.kaggle.com/competitions/name-that-llm/data) · [官方比赛说明](https://www.kaggle.com/competitions/name-that-llm/overview/description)

## 优先实验

| 顺序 | 单因素比较 | 实施与判断重点 |
| --- | --- | --- |
| 1 | **精细风格特征**：E0c 对照 E0c＋拆分 Unicode 标点、功能词/词性 n-gram | 现有 57 项特征把直/弯引号及不同 Unicode 破折号合并。先分别加入具体字符频率，再试只保留功能词和词性序列的稀疏特征；每组先单独比较 E0c，再按原固定规则与 E2d-R 融合。检查是否改善同家族版本召回，而非只提高家族识别。[原始模型指纹研究](https://arxiv.org/abs/2405.14057)发现字符、词和词性线索有效，但不保证本赛增益。 |
| 2 | **截断审计与长度消融**：RoBERTa 256 对照 512 tokens 或首尾拼接 256 tokens | 先在开发文本上记录当前分词器的总截断率及逐类截断率；若比例可观，再用相同五折、12 轮和有效 batch 大小重训。分别报告单模型与固定 E0c 融合的 OOF。不要仅凭更长输入推断更好。 |
| 3 | **融合分数**：现有等权排名对照逐行标准化分数后等权融合 | 先试无需拟合的逐行分数标准化。若拟合温度、校准器或融合权重，必须在每个外层验证折的训练 prompt 内另做分组交叉拟合；不能直接在同一批 OOF 标签上拟合并报分。scikit-learn 多分类校准默认使用 StratifiedKFold，不能直接当作本赛的按 prompt 切分。[校准器文档](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) |
| 4 | **同家族软专家**：直接 18 类分数对照小权重家族内专家融合 | 根据开发 OOF 混淆矩阵预先选少数易混版本对，只训练对应二分类专家；仅对相关类别分数作连续、小幅调整。已有硬式“先家族后版本”E7a 得分 0.3250，低于直接 E0c 的 0.3426，故不采用硬路由。 |
| 5 | **经过审计的外部数据**：官方数据对照官方＋公开语料 | 逐来源核实真实 `served_model`、路由、许可证、prompt 来源及与官方数据的近重复；先做每类和每来源限额，再比较少数预定权重。已有 E3 的 2,000 条/8 类外部数据无净收益，更多行数本身不足以证明有效。[公开逐行来源数据卡](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations) · [外部数据审计](external_data_audit.md) |

较重的后续路线是**来源条件化似然**：ATLAS 论文让模型分别估计各来源文本的似然，并裁剪单个 token 的证据，以提高分布变化下的归因稳定性；论文的跨领域、跨版本结果不能直接当成本赛收益。[ATLAS 原论文](https://arxiv.org/abs/2610.19583) 当前一份公榜 0.667 的[公开 Notebook](https://www.kaggle.com/code/lomnafsk/ntl-triple-margin0-7)结合多个 8B/14B LoRA、额外生成数据、n-gram SVM 和少量 ATLAS 分数。这说明高分方案可能需要来源可靠的全类别数据与互补模型，但成本远高于前述单因素消融，先不照搬其多路融合。

## 共同评估与停止条件

- 沿用现有 **120 个开发 prompt、五折 GroupKFold** 及同一批 2,160 条 OOF ID；每折只在训练 prompt 上拟合词表、标准化、校准及专家。比较 pooled balanced accuracy、五折变化、18 类召回率与混淆矩阵，并按 prompt 做配对 bootstrap。多次候选选最高者会使 OOF 乐观，微小或不稳定的增益仅记为探索性结果。
- 30 个 prompt 的保留集已执行一次最终检查，**不再用于筛选、调参或重新估计**。新候选若开发集证据充分，应另训全量模型；Kaggle 公榜仅作为分布偏移信号，不用反复调参，保留原提交作对照。
- 每个晋级候选还要在断网 Notebook 中通过约 4,000 行的模拟规模推理、`submission.parquet` 格式检查和九小时运行限制。[官方代码赛要求](https://www.kaggle.com/competitions/name-that-llm/overview/description)

## 暂不优先

E4 的同题难负例 SupCon 未超过交叉熵对照。也不建议对推测出的测试 prompt 强制执行“每类仅出现一次”的匈牙利匹配：一位参赛者报告该方法将本地 OOF 从约 0.374 提至 0.481，但公开榜单从 0.353 降至 0.350；这只是该参赛者的观察，足以提示迁移风险。[原始实验帖](https://www.kaggle.com/competitions/name-that-llm/discussion/751802)
