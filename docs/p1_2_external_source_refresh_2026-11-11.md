# P1-2：新外部回答来源复查（2026-11-11）

本轮只查公开数据卡、仓库文件清单及生成协议，没有下载回答、训练模型或改动 E24/E30-B 协议。准入沿用 [E24 行级审计](e24_prvsiyan_source_audit.md)：原请求、逐字提示、原响应、实际服务模型和供应商、路由回退、完成状态须能逐行核对；来源的生成与筛选不能预先利用本项目外层验证 prompt 的回答或标签。E31/E32 的联合严格准入为 0 行，详见[独立来源审计](e31_independent_source_audit.md)和[原始调用审计](e32_willrice_raw_audit.md)。

| 新查来源 | 已能核实的公开信息 | 严格五折结论 |
| --- | --- | --- |
| [seyitkaangunes：Extra API Passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) | [Kaggle 公开元数据](https://www.kaggle.com/api/v1/datasets/view/seyitkaangunes/ntl-extra-api-passages) 于 2026-11-11 返回文件版本 **2**、CC BY 4.0；v2 注释称与 v1 **同一批数据**，只修订 README。发布者称五个 Claude 标签共 4,998 条、1,000 个提示，第一方 Anthropic API 返回 ID 与请求一致。公开描述仅列 `ntl_extra_passages_claude.csv` 和 `prompts.csv`，回答表列中无原请求/响应 envelope、逐行返回模型、调用时间及响应 ID。提示曾按**全部官方训练主题**过滤，目标字数从全部官方训练回答取样。发布者还用官方训练数据训练风格分类器并在同一官方训练集的分组 CV 上调过外部数据权重；其报告的增益不能替代本项目预登记实验。 | **当前可独立复核身份的行数未证实，严格准入 0 行。** 即使作者补齐调用记录，既有全训练集参与的提示筛选仍限制严格五折解释。不能把发布者的自述当作逐行核验。 |
| [Tadeu Hayashida：Claude Fable 5](https://www.kaggle.com/datasets/tadeuhayashida/ntl-claude-fable-5-generations) | [Kaggle 公开元数据](https://www.kaggle.com/api/v1/datasets/view/tadeuhayashida/ntl-claude-fable-5-generations) 返回 v1、CC BY 4.0；发布者称 1,484 条清洗回答与 1,641 条三种模板的原始回答，字段包括返回模型、`response_id`、时间、变体和系统提示。提示来自第三方库，已按**官方训练提示主题**过滤；`sys` 模板被选入清洗文件，是因其标点分布匹配官方训练回答。清洗还比较了可见测试文本和公共池。 | **不进入严格五折。** 原始调用 envelope 是否存在仍未核实；即使存在，模板选择和提示过滤已接触外层验证信息，且只有一个类别。原始 `base` 100 条可能供描述性诊断，但不代表独立比赛协议。 |
| [RohanExploit：extra GPT/Gemini generations](https://github.com/RohanExploit/kaggle-name-that-llm-top20/blob/main/data/extra_gpt_gemini/README.md) | 仓库公开了 `ntl_generations.csv`、`prompts_used.csv`、`make_tasks.py` 和 README；README 称使用 Kaggle Benchmarks 代理，CSV 的 `model_seen` 为代理返回的模型字符串，覆盖 5 个 GPT 与 3 个 Gemini 标签，CC BY 4.0。目标字数按**每个官方训练标签**的字数分布抽样；README 所列文件没有原始请求/响应 envelope、代理真实后端/供应商或禁回退证明。另一团队的[2,503 行审计与 E0c 增益](https://github.com/kolihhan/name-kaggle)是其自己的报告，不能证明这些代理行的真实服务快照或本项目的严格独立性。 | **不进入严格五折。** `model_seen` 不能从代理 slug 推出底层精确版本；全训练标签分布也参与生成。 |
| [TheShelter：shelter-observatory](https://huggingface.co/datasets/TheShelter/shelter-observatory) | 独立于本赛、CC BY 4.0；HF 页面显示 2,553 条回答、20 个模型键、**只有 13 个不同提示**，含 `requested_model`、`served_model`、供应商、时间、提示和回答；可见 Qwen 3.8 Max 行。英文与西班牙文混合，提示集中在 AI 自我叙述，回答长短跨度约 29–2,190 词。页面列出的字段来自整理表，未见原始 API envelope 的逐行核验。 | **不做 E0c 训练。** 13 个高度重复的专门提示和部分超长/西语回答与本赛 40–250 词、多学科英语单段任务有明显域差；它也无法证明比赛 Qwen 隐藏回答采用 0902 快照。可作为非训练的路由字段示例。 |

**决定：没有发现一个新来源同时满足独立生成/筛选与 E24 式逐行身份核验，本轮不给 E0c 或 GPU 训练新增外部行。** 这不否定公开回答本身；只是现有证据不足以把来源自带标签当成比赛 18 个精确快照。E24 的已审计清单、E29 Qwen 0902 敏感性负结果和 30-prompt 保留集边界保持原样。

后续只读核查见 [P1-3 Oleg 来源预审](p1_3_oleg_corpus_provenance_audit.md)：其 18 类外部回答语料虽然规模大，Speechmap 和 Woog 的来源目录声称有请求及返回模型，但原始逐次请求/路由日志或输出许可尚不足以通过本项目门槛，严格准入仍为 0 行。另一个独立提示线索 [Model Spec Evals](https://github.com/Michel-Justen/Model-Spec-Evals) 声称保留原始 `.eval` 日志；仓库未声明生成回答许可证，[请求检查脚本](https://github.com/Michel-Justen/Model-Spec-Evals/blob/main/src/verify_requests.py)默认仅查 5 条且不核查逐条返回模型，[导出脚本](https://github.com/Michel-Justen/Model-Spec-Evals/blob/main/src/export_reasoning.py)使用命令行标签和日志头模型名，故当前同样为 0 行。原始提示的 [CC0 许可](https://github.com/openai/model_spec_dataset)不能自动覆盖后续生成回答。

## 若取得原始证据，先做的小规模筛查

1. 在看任何验证分数前冻结公开版本、文件 SHA、许可证、独立提示许可与每条 `call_id → 原请求/逐字提示/路由参数 → 原响应模型/服务商/回答/停止状态` 映射；对 Qwen 等别名另记比赛目标快照未知。原始生成或筛选若使用了全部官方 150 个提示/2,699 条回答的标签驱动信息，不能声称严格独立五折。
2. 只在既有 **120 个开发 prompt、五折 GroupKFold** 下比较官方 E0c 与“官方 + 单一新来源”E0c；每类外部总权重预先固定为该类官方折训练权重的 0.25，缺类不补别的来源。各折词表、标准化和分类器只拟合该折训练行加固定外部行；30 个保留 prompt 不参与选源、去重或调参。
3. 预先固定排重及来源域检验规则，报告全五折 pooled BA、逐折、逐类召回、按 prompt 配对区间和去掉该来源后的敏感性；先确认新来源对 E0c 有稳定增量，再考虑与 E24 的固定融合。若原始 envelope 或生成前协议缺失，则只做描述性诊断，不把同一开发集上的分数写成严格新数据增益。

最有价值的下一步是向上述 Claude 数据作者索要**未经过全官方训练数据筛选的提示/调用清单和原始 API envelope**；对 Kaggle proxy 来源还需能独立核对真实服务后端。若打算自行生成，应先锁定独立提示和第一方固定快照调用协议，记录完整请求/响应并核对服务条款与费用。
