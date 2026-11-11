# P1-4：Open Pool v16 与上游来源预审

审计日期：2026-11-11。只读取公开 Kaggle 数据卡、版本元数据、文件清单及小型 README／统计文件；未下载回答表、训练模型或使用保留集。准入沿用 [E24 行级审计](e24_prvsiyan_source_audit.md)：原请求与逐字提示、原响应、实际服务模型和供应商、路由及回退、完成状态均须逐行核对；外部来源的生成和筛选不能预先利用外层验证 prompt 的回答或标签。

## 254k v7 线索

讨论中的“Trust-Worthy Pool Dataset - 254k free passages for all 18 models (v7)”指向既有的 [crtproduction/name-that-llm-open-pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)，不是新的独立来源。[Kaggle 公开版本元数据](https://www.kaggle.com/api/v1/datasets/view/crtproduction/name-that-llm-open-pool)在审计时返回**文件版本 16**、更新于 2026-11-09、CC BY-SA 4.0；页面标题为“224k Clean Passages”，卡片正文仍以内部 **v14／223,840** 条为题。254k 是旧版清洗前规模，不能当成当前可准入行数。

新增的 `gen_route`、`gen_served_model`、`source_protocol.csv` 等列汇总不同发布者的记录，不等于每行原始 API 请求和响应。池的 `match_weight` 取自全部官方训练回答的排版统计，`fidelity` 分析及部分来源／路由取舍也使用了全部官方训练答案和标签；主文件不能反推未筛选的原始来源。沿用 [既有来源审计](e24_prvsiyan_source_audit.md)，**池主文件新增严格准入 0 行**。这只是本项目五折协议下的结论，不断言每条上游原始回答都无价值。

## 四个上游线索

| 上游公开来源 | 本轮直接核对 | 严格 E0c 准入 |
|---|---|---|
| [kilupy：Claude-family passages](https://www.kaggle.com/datasets/kilupy/ntl-claude-family-passages) | [元数据](https://www.kaggle.com/api/v1/datasets/view/kilupy/ntl-claude-family-passages)为文件 v2，2026-11-02 更新，Kaggle 许可证栏为 **Unknown**；公开 `README.md` 声称 CC BY 4.0、官方 Anthropic API、独立编写的 3,338 个提示及 16,642 条回答。文件清单只有 README、提示生成脚本、回答 Parquet、提示 CSV 和统计 JSON；回答字段有提示、文本、用量和延迟，**没有返回模型、响应 ID、完成状态或原始调用 envelope**。README 称每个提示都由五个模型回答，但 `stats.json` 的逐模型计数合计 16,642，较 3,338×5 少 **48** 个模型—提示单元。 | **0 行**；这是最值得补证的来源，但自述标签不能代替原始调用核对。 |
| [davidpiln：Vertex Gemini passages](https://www.kaggle.com/datasets/davidpiln/ntl-vertex-gemini-gen) | [元数据](https://www.kaggle.com/api/v1/datasets/view/davidpiln/ntl-vertex-gemini-gen)为 v2、CC BY 4.0；卡片称 4,900 条，逐行保存 API 返回的 `model_version` 与 `finish_reason`。其中 4,600 条 A/C 组提示虽新写，却以重构的官方训练提示作示例，并按**全部**重构训练提示相似度筛选；B 组复用其他来源的提示。公开文件只有 README、回答 CSV 和提示 CSV，没有原始 envelope。 | **0 行**；原始日志也无法消除生成前已触及全部训练提示的五折污染。 |
| [elliotthankinson：exact-identity gens v3](https://www.kaggle.com/datasets/elliotthankinson/ntl-exact-gens-v3) | [元数据](https://www.kaggle.com/api/v1/datasets/view/elliotthankinson/ntl-exact-gens-v3)为 v7、CC BY 4.0。来源混合 Kaggle 代理及 OpenRouter；卡片明示用全部比赛训练回答做近似过滤，并用其训练的判别器筛选模型/模式单元。公开文件清单为 README、费用说明和若干 Parquet，未见原始调用 envelope。 | **0 行**；既有筛选已利用外层验证答案。 |
| [zeminglii：own exact-route generations](https://www.kaggle.com/datasets/zeminglii/ntl-own-exact-route-generations) | [元数据](https://www.kaggle.com/api/v1/datasets/view/zeminglii/ntl-own-exact-route-generations)仍为 v2，非本轮新增版本；提示取自 Open Pool，目标字数按全部官方训练标签分布抽样。公开文件仅 README 和回答 CSV，无原始调用 envelope；[E12 审计](e12_external_data.md)已有逐表核查。 | **新增 0 行**；旧来源条件未改变。 |

## 下一准入门槛

优先向 kilupy 核实并获取每条回答的**原始、可关联**请求与响应记录：逐字用户消息、请求模型、实际返回模型、响应 ID、供应商／路由、HTTP 与完成状态、生成时间、文本，以及失败调用清单；同时请其将 Kaggle 元数据许可证与 README 的 CC BY 4.0 对齐。取得后先核对文件版本、哈希、提示独立性、缺失的 48 个配对单元和逐行身份，再按 [P1-2 固定 E0c 门槛](p1_2_external_source_refresh_2026-11-11.md)预登记来源、抽样和权重。**当前四个线索均未向严格五折新增训练行，不启动基于它们的 GPU 实验。**
