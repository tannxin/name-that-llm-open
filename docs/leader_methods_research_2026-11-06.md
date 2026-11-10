# Name That LLM：高分公开方案与往届方法（2026-11-06）

2026-11-10 新增的 0.680 公开方案复核与后续实验优先级见[更新记录](leader_methods_update_2026-11-10.md)。

## 证据边界

本赛尚未结束，没有最终获奖方案或私榜结果。以下本赛分数均为公开 Notebook 页面显示的公榜。我们的 E10 公榜为 **0.401**，E16 公榜为 **0.407**；E16 的 **0.409259** 是开发集按 prompt 分组的 OOF，不能直接与公榜相减。E24 开发 OOF 为 **0.425000**，其 Kaggle 隐藏重跑已于 2026-11-10 提交，公榜分数待出。官方训练集 2,699 条、150 个 prompt；隐藏测试约 4,000 条，prompt 未在训练集出现；指标为 18 类平均召回率，公榜约占测试行 40%，公私榜类别比例不同。[规则与指标](https://www.kaggle.com/competitions/name-that-llm/overview/description) · [数据说明](https://www.kaggle.com/competitions/name-that-llm/data) · [我们的结果](results.md)

## 本赛公开材料

| 来源 | 页面可核实的内容 | 不能据此断言 |
| --- | --- | --- |
| [0.666 Notebook V47](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) | 页面显示公榜 0.666；完整公开源码可读。其核心含多视角稀疏分类器、多个公开模型读出、互补模型栈、条件家族门控、ATLAS、Qwen3.8-27B LoRA、边际校正和排版证据。展示的 21 秒运行只命中三行公开样本的固定预览分支，跳过模型前向。 | 0.666 是 27B LoRA 的单独收益；21 秒是隐藏测试推理或训练耗时；公开展示运行足以验证约 4,000 行的隐藏执行。 |
| [0.600 Notebook](https://www.kaggle.com/code/kirilltym/lora-adaptive-claude-gates) | 公榜 0.600；23 个输入，包含外部 API 文本、Qwen3-8B LoRA 与其他模型资产。 | 标题中的去 prompt 方法有多少独立收益。 |
| [公开扩展池](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) | 当前标题为 “213k Clean Passages”；发布者称覆盖 18 类。 | “Clean” 等于逐行真实服务模型、路由及授权可靠；旧版 254k 数字等于当前版本行数。 |
| [7,753 条 API 数据](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations) | 数据卡列出 provider、served_model、api_route、prompt、长度、来源等字段；覆盖 10 类。 | 缺 served_model 的行或自动低价路由行可直接视为官方模型同分布样本。 |
| [Hungarian 匹配反例](https://www.kaggle.com/competitions/name-that-llm/discussion/751802) | 参赛者报告按文本聚类并强制组内类别互异，使 OOF 从 0.374 到 0.481，公榜却由 0.353 降到 0.350。 | 训练集“每 prompt 每模型一条”的结构在测试集仍成立。 |

我们已通过 Notebook 正文 iframe 逐段核查 V47 源码。我们的 0.401 与其 0.666 相差 **0.265 公榜 balanced-accuracy 点**。该方案结合大量外部数据、他人公开权重和自己的增量，页面没有逐组件消融，不能把差额归因给标题中的 27B。先前 E3 只用 8 类约 2,000 条外部数据，E12 大池又因来源问题被拒，两者都不能视为已验证过“高质量 18 类扩展训练”。

### 0.666 V47 源码核查：可借鉴的机制与归因界限

- **数据与稀疏主干。** Notebook 自述将 Carrot Production Open Pool 数据发布 v11（Kaggle Dataset v13）的 **190,588 条**清洗文本用于四视角 SVC，并按类别用 `match_weight` 抽取外部样本拟合 TF-IDF 与文本失真逻辑回归。三路在核心分数中的权重分别为 **12、0.25、0.10**；再与 Qwen3 14B／8B、Llama 3.1 8B 的 V5 LoRA 多层读出、DeBERTa 和同题 Gaussian 读出合成。源码的旧缓存清单含 24,443 条匹配去重样本；拟合后的数值状态存于附加 Dataset，页面不能独立证明外部样本每行确由所标精确模型与路由生成。此处的 v13／190,588 是该提交锁定版本，不能替换成当前 Dataset 标题的行数。[V47 源码](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) · [Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)
- **四视角的确切定义。** 公开 `svc4_runtime.py` 将原文字符 3–5 gram（最多 250,000 特征、块权重 1.0）、原文词 1–3 gram（180,000、0.7）、shape 字符 2–6 gram（60,000、0.5）、功能词 1–3 gram（50,000、0.5）拼接。四路 TF-IDF 均设 `lowercase=False`、`sublinear_tf=True`、`min_df=2`；词视角的 token pattern 包括标点，功能词视角将非功能词替换成 `CONTENT` 并保留标点。shape 的预处理是 `re.sub(r'[^\W\d_]+', lambda m: 'X' if m[0].isupper() else 'x', text)`，仅当整段字母均大写才用 X 表示（m[0] 是整段匹配），其余数字、标点和空白保留。拼接后为 `LinearSVC(C=0.5, class_weight='balanced', max_iter=5000, random_state=42)`。这些是其预拟合工件的读取契约，不能由此证明 shape 单路的增益；我们的最小消融只在原 E0c 添加这一块，保留其余训练配置。[V47 源码](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) · [下一轮计划](next_round_plan_2026-11-06.md)
- **融合而非单模型。** 有效代码先将核心与互补栈概率按 **85/15** 混合；仅对前两名概率差小于 0.3 的行混入 **10% ATLAS**；再对所有行混入 **5% 自训 Qwen3.8-27B GSQ3 LoRA**。前置配置一度写 7%，但最后一单元将 `gsq_share` 改为 5%，执行入口也断言 5%。Notebook 自述这一路先用 119,899 条公开语料训练，再用官方 2,699 条训练数据两轮适配；这是作者及工件元数据的训练报告，未找到可独立核实的逐步训练消融。[V47 源码](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313)
- **软家族与后处理。** 互补栈的 Claude、GPT、Gemini、GLM 门控只在家族内按置信度重排兄弟模型，并以 `logsumexp` 保持原家族总概率。后处理将预测边际的 65% 与平滑官方训练先验的 35% 作为目标，以 0.30 强度做软平衡，再加 37 个标点／排版标记的分层 Beta-Bernoulli 对数证据；证据按置信度、文章长度及标记覆盖率缩放。源码明确无最终硬标点遮罩或 Hungarian 分配。[V47 源码](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313)
- **来源与运行边界。** Notebook 标注继承 Roman Tamrazov 的父集成、mercurydust 的 0.660 V2 集成及排版方法、Joseph Adamski 的 ATLAS；作者还称 Areeb Ali 的 Open Pool／SVC V14 已获 0.666，且**没有**此 Qwen 支路。这是 Notebook 的来源说明，尚非我们独立核实的 Areeb V14 分数；无论如何，这份 Notebook 的 0.666 公榜无法证明 Qwen 支路增加了分数。公开 21 秒版本在测试输入恰好为三条带固定 ID／指纹的样本时直接写预置预测，`if not preview` 才运行完整模型。比赛官方说明公开交互测试仅有样例，评分时重跑约 4,000 条隐藏测试；可见运行耗时和输出不能作为完整推理的资源测量。[V47 源码](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) · [官方数据](https://www.kaggle.com/competitions/name-that-llm/data)

可执行借鉴按成本排序：已做**一个固定参数的 shape 块消融**，单路 +0.0111、固定 E16 融合仅 +0.0042（[E20 结果](e20_shape_ablation.md)），低于预设晋级门槛；下一步在可核验来源数据上测试增量；守恒家族概率的软门控、低强度排版证据和边际校正可分别用现有 OOF 做低成本消融。E0c 已含词／字符 TF-IDF 与 57 个风格特征，精确标点率和功能词 1–3 gram 的直接拼接在 [E8](e8_fingerprint_ablation.md) 中增益小且区间跨零，不能当作未做过的新实验。弱标签 Open Pool 可另行探索，但须去掉依赖全部官方训练集的 `match_weight`／`style_score` 等列、记录来源噪声与提示选择偏差，且不能把其结果冒称精确版本数据收益。只有低成本消融或互补性明确后，才考虑 8B LoRA、多层读出及 27B。每项使用官方 `prompt_id` 五折的训练侧拟合和同一验证侧比较，融合权重不能照搬上述数值。[下一轮计划](next_round_plan_2026-11-06.md) · [E12 审计](e12_external_data.md)

## 往届可核实的方法

SemEval 是“人类＋五个模型”的六分类，报告准确率；DAIGT 是“人类／机器”二分类，报告 AUC。两者的分数都不能直接与本赛 18 类 balanced accuracy 比较，下面只借鉴可检验的训练与验证设计。

| 队伍与原始来源 | 核实的方法与结果 | 对本赛的优先级和边界 |
| --- | --- | --- |
| **SemEval-2024 Task 8 B 冠军 AISPACE**：[论文](https://aclanthology.org/2024.semeval-1.212.pdf)、[官方成绩](https://aclanthology.org/2024.semeval-1.279.pdf) | 将 B 任务的 71,027 条与 A 任务可重标注数据合并、去重并排除开发集重合行，得到 127,755 条。相同 RoBERTa-large 设置下，开发准确率 **0.739→0.905**；加类别加权交叉熵到 **0.915**；调轮数和学习率后到 **0.980**。按逐类表现选择 RoBERTa、DeBERTa、XLNet、Longformer 进行概率软投票，开发集 **0.9946**，跨域正式测试 **0.9085**。 | **P0：高可信同标签数据的增量与去重消融。P2：以 OOF 逐类互补性选择异构软融合。** 数据分布变化造成的开发／测试差距很大；本赛类别近乎均衡，类别加权损失应单独验证，不能默认有效。 |
| **SemEval-2024 Task 8 B 亚军 UniBuc**：[论文](https://aclanthology.org/2024.semeval-1.63.pdf)、[开源代码](https://github.com/ClaudiuCreanga/semeval-2024-task-8) | 六分类最终模型为 RoBERTa-base 最后一层 `[CLS]` 加分类头；原始文本优于轻／重清洗。先冻结骨干训练分类头 6 轮，再仅解冻骨干最后一层训练 2 轮；正式测试准确率约 **0.8695**。 | **P1：在同一分组折上比较冻结、逐层解冻和全量微调，保留标点格式。** 论文的截断结论来自较长文本，不直接决定本赛短文本的输入长度；其字符 CNN／BiLSTM 主要用于另一个边界检测子任务，不是六分类亚军模型。 |
| **Kaggle DAIGT 冠军**：[获奖复盘](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/comprehensive-1st-place-write-up)、[冠军代码](https://github.com/rbiswasfc/llm-detect-ai) | 构建覆盖多生成器、提示、采样设置和改写方式的数据；最佳单模型约用 **16 万条**训练样本。模型包括 Mistral-7B (Q)LoRA、DeBERTa-v3 分类／排序、Ghostbuster 式 token 似然特征；最终平均各模型的预测排名，以适应 AUC。代码仓库还提供生成、训练和对比学习 embedding 脚本。 | **P0：优先改善来源、生成协议和长度覆盖；P2：检验词／字符／似然特征与神经模型的互补性。** 二分类 AUC 下的排名融合不适用于直接输出本赛 18 类标签；改写增强可能抹去来源指纹，需单独消融。仓库使用 4 张 A100／A6000 训练，不能把其资源成本等同于一次 Kaggle 推理。 |
| **Kaggle DAIGT 亚军 Guanshuo Xu**：[获奖复盘](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/guanshuo-xu-2nd-place-solution-with-code-and-data)、[推理 Notebook](https://www.kaggle.com/code/wowfattie/daigt-2nd-place)、[训练代码与数据](https://www.kaggle.com/datasets/wowfattie/daigtpretraindata) | 早期 DeBERTa 在共享数据上 CV 近满分，公榜却偏低且不稳。随后从 SlimPajama 用多种 LLM 产生约 **50 万组**人类／机器文本，先训练通用 DeBERTa-v3-large，再用贴近目标学生作文的生成数据适配；最终融合两份 DeBERTa-v3-large 与一份 DeBERTa-large。TF-IDF 融合提高公榜但降低私榜；最终私榜 AUC 约 **0.983**。 | **P1：比较外部池预训练→官方折适配与混合一次训练，并设 prompt 未见的锁定保留集。** 其人类／机器配对不能直接变成本赛 18 个精确模型标签；外部生成文本须逐行核实真实服务模型和路由。 |

共同教训是先验证数据协议和未见 prompt 的泛化，再投入更大的骨干与复杂损失。DAIGT 冠军使用的测试集 MLM、伪标签和按公榜直觉定融合权重属于该赛特定做法；本赛若考虑这些方法，须先核对当前规则并在锁定 OOF 上独立证明收益。

## 下一轮建议顺序

1. **审计并锁定外部数据版本，明确两条路线。** 正式训练数据逐源核对许可证、实际服务模型 ID、代理路由、提示来源、18 类覆盖、生成协议及近重复，分别记录接受、存疑和拒绝行数。另可把旧 E12 约 212k Open Pool 等汇总池作为**弱标签探索**，但禁用用全部官方 train 算出的 `match_weight`、`style_score` 等字段，记录可能的来源误标和提示选择偏差并做来源剔除敏感性分析；不能因为标题写 “clean” 就视为已核实的精确标签。[E12 审计](e12_external_data.md)
2. **先用便宜模型测数据价值。** 保持现有 120 prompt 五折 GroupKFold，比较官方数据单独训练与“官方＋高可信外部行”训练的 E0c；弱标签池另列为探索性对照，不混入正式来源增益。预先固定少量来源权重，报告每类召回、逐折与按 prompt 配对区间。已做过的精确标点和功能词特征拼接不重复扫参，新增稀疏实验聚焦分离视角、shape 与软证据的独立贡献。[E8](e8_fingerprint_ablation.md) 外部数据的去重与验证 prompt 排除规则应在看分数前固定。原有 30-prompt 保留集已使用一次且进入全量训练，不能再当作未使用的锁定测试集。
3. **数据消融通过后再做两阶段神经模型。** 用高可信外部池训练 DeBERTa-v3-large 或 ModernBERT，再在每个官方训练折上适配，保留完整 OOF。比较仅官方训练、混合训练、外部预训练后官方适配；在相同折上补充“冻结骨干→解冻顶部层”对照，尤其看同家族版本召回是否改善。
4. **最后试 Qwen 8B LoRA 和低自由度融合。** 先做单折或小规模消融与推理时限检查，证实互补性后才扩大到 27B。融合只用对齐的官方分组 OOF，保存每路 18 类分数并按本赛 balanced accuracy 选权重；不要照搬二分类 AUC 的排名平均，也不要按公榜反复调权重。
5. **暂缓测试时硬分组和复杂损失。** 官方测试没有 prompt_id，公开 Hungarian 反例说明训练集的一对一结构不能直接外推。若试软式去内容化，先在未见 prompt、不同主题和长度分层上证明稳定收益。

晋级仍需多数折和类别方向一致、重要类别召回没有严重倒退，并用少量公榜提交做外部核验。E16 公榜 **0.407**；E24 于 2026-11-10 提交，隐藏重跑正在进行。[E24 提交交接材料](e24_kaggle_handoff.md)
