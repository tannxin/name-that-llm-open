# E24 原始外部来源审计与准入清单

审计日期：2026-11-06。此文只锁定 E24 的外部来源和行级准入；五折效果由独立实验记录报告。原始第三方文本、API envelope 和准入 manifest 均留在 Git 忽略的 `data/external/`，不上传到 GitHub 或 Kaggle。

## 为什么不用 Open Pool v15 主文件

[crtproduction 的 Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) 当前 Kaggle 文件版本 **15**、数据卡内部 **v13**，公开、CC BY-SA 4.0，卡片称主文件有 **212,623** 行、18 类、60 个来源。本地保存了公开 API 返回的 `data/external/crtproduction/kaggle_metadata_v15.json`，没有把主文件用于拟合。

该数据卡的 **“Why clean: the evidence” 第 2 节**写明，作者比较原始与清洗池时，外部池训练的分类器随后在全部 **2,699** 条官方训练回答上评分；原文关键句是 “It was then scored on all 2,699 competition train passages”。其第 2–4 节据这些分数采用垃圾文本、偏离体裁来源及许可证/来源过滤。**第 6 节 “Tests behind v11” 的表格**明确以 **“All 150 train prompts”** 为一列评估 Gemini 3.7 的路由来源选择，并将仅留 Vertex AI 行的决定标为 “Adopted: the main file now keeps only documented Vertex AI rows”。同一数据卡还说明 `match_weight` 用完整官方训练集的排版和长度分布拟合，`style_score`、`max_cos_train` 等也依赖官方训练集。[原始数据卡与这些位置](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)

因此，即使完全忽略上述衍生列，主文件**已有**按官方开发验证 prompt 的答案及标签进行的来源/路由选择。仅凭 v15 主文件不能反推未筛选的原始来源，并重做只看各外层训练折的清洗。其 `model` 也是原作者汇总的标签，主文件没有逐行实际返回的服务模型 ID/服务商/路由。**Open Pool v15 主文件对严格的官方 prompt 五折外部数据消融判为 no-go**；不报告它的五折改进，也不把它当作精确版本标签。

## 改用 prvsiyan v9 的固定原始 split

[prvsiyan 的 Extended Name That LLM Dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) 公开可匿名下载，Kaggle 文件版本 **9**，数据卡许可证 **CC BY 4.0**。下载脚本 [download_prvsiyan_v9.py](../download_prvsiyan_v9.py) 从带 `datasetVersionNumber=9` 的公开单文件 API 获取 `passages.csv`、`prompts.csv`、`provenance_summary.json` 和 `provenance.jsonl`；文件 SHA-256 由 [audit_prvsiyan_v9.py](../audit_prvsiyan_v9.py) 锁定。本地版本卡与下载 manifest 位于 Git 忽略的 `data/external/prvsiyan/`。

只读取 raw `passages.csv` 的 **`split=new`**：300 条为本来源写的提示，每条由 18 个标签各回答一次，合计 **5,400 行**。`prompts.csv` 中每条 `new` 提示的正文、模板及 CC BY 4.0 提示许可均存在；5,400 行逐行渲染后的 `prompt_sent` 与模板及目标字数一致，`source_prompt_id` 全为空。绝不使用 `train_regen`（重构的官方 150 个训练提示）、按这些提示风格编写的 `new2`、`proxy_v1`、`pilot_range` 或 `new_kimi_nothink`。

**不用** `passages_clean.csv`、`passes_funnel`、`passes_funnel_regen`、`max_shared_8grams_comp`、`train_vs_dataset_stats.csv` 等字段或文件来选样；来源的 clean funnel 包含与**全部官方训练回答**的 8-gram 比较。原始 5,400 行自身全部非空、正常 stop、未标记拒答/截断，大小写折叠并归一化空白后的回答无内部重复。

## 原始 API envelope 与路由审计

完整 `provenance.jsonl` 为 **103,099,765 字节**，SHA-256 `a7e78ff29ba2510fc9180cfcf864e0a1ecc1c36ee1bbd059785b2ad95b2a94de`，可解析 **13,213** 条调用记录。`new` 的 **5,400** 个 `call_id` 各唯一命中一个原始 envelope。对全部 5,400 行逐项核对 CSV 与原始请求的 `prompt_id`、标签、路由、请求模型、逐字提示，以及原始响应的实际模型 ID、服务商、回答文本、响应 ID、完成状态；HTTP 200、身份检查 `ok` 和服务商固定/关闭回退约束也通过。**缺失、重复和字段不一致均为 0**。Google Vertex 使用响应的 `model_version`，其余路由使用响应的 `model` 字段，而非仅看请求名或汇总标签。逐行结果可由 [审计脚本](../audit_prvsiyan_v9.py) 重算，摘要在 Git 忽略的 `outputs/e24_prvsiyan_audit/report.json`，`audit_passed=true`。

准入路线只收第一方 Anthropic/OpenAI/Google Vertex，及 OpenRouter 明确固定原厂供应商且禁止 fallback 的 Z.AI 和 Moonshot AI；所有行仍需通过上面的原始响应核验。排除 **730** 条 Kaggle proxy 行：代理仅回显模型 slug，没有后端实际版本字符串；排除 **48** 条 OpenRouter Gemini 行以保持 Google 版本只由 Vertex 的 `model_version` 证实。另排除 **300** 条 Qwen：它们的请求和返回 ID 均为 `qwen/qwen3.8-max-0902`、供应商 Alibaba，但比赛标签 `qwen3.8-max` 的目标快照未获证实。Qwen 可供单独的版本敏感性探索，当前不进入精确标签消融。

| 准入类别 | raw `new` 行 | 逐行路由合格行 |
| --- | ---: | ---: |
| Claude 五类，各类 | 300 | 300 |
| Gemini 3.1 Pro / 3.6 Flash / 3.7 Flash / 3.8 Flash | 各 300 | 215 / 216 / 300 / 216 |
| GPT 五类，各类 | 300 | 195 |
| GLM 5.3 / 5.3 Flash，各类 | 300 | 300 |
| Kimi K3 | 300 | 300 |
| Qwen 3.8 Max | 300 | 0 |
| **合计** | **5,400** | **4,322（17 类）** |

`gpt-5.5` 的第一方实际返回 ID 为 `gpt-5.5-2026-04-23`，与来源数据卡的明示快照相符；比赛只给别名，仍存在目标快照未知的迁移风险。其他别名也可能随服务时间漂移，不能把行级服务身份等同于证明与主办方生成时的隐藏配置完全相同。

## 近重复与固定实验边界

只取官方 split manifest 的 **120 个开发 prompt、2,160 条回答**做无标签近重复检查；30 个 holdout prompt 的答案和标签未参与相似度或准入决策。将全部 5,400 条 `new` 回答与开发回答一起拟合 char 4-gram TF-IDF（`min_df=2`、`max_features=100000`、`lowercase=True`），每条外部回答对开发回答的最高余弦中，整体最大值 **0.35484**；≥0.5、≥0.8、预定排除阈值 **≥0.90** 都是 **0 行**。另外，300 条 `new` 的 `prompt_body` 对开发 120 条社区重构提示的 `prompt_body` 做英文停用词过滤的 word 1–2 gram TF-IDF，最大余弦 **0.09656**。这些检查不能证明与未公开的真实提示在语义上完全不同，也不能覆盖隐藏测试。

准入 manifest 为 `data/external/prvsiyan/e24_accepted_manifest.csv`，**4,322 行、17 类**，SHA-256 `99c1abc12b64c7a0b1bfada6393383c4353c776c69105cecf68bee7e08439a7a`，含源行 ID、prompt ID、标签、call ID、路由、实际服务模型及服务商、原 CSV SHA。该清单已完整审计，但还未按实验限额抽样。E24 已预登记：对 300 个外部 `prompt_id` 以固定 SHA 哈希排序取前 250 个；每个准入类别只取这些 prompt 下的行，缺行不补位。每折每类 96 条官方训练回答权重 1；若外部取到 `n_c` 行，则每条外部权重 **`24/n_c`**，让该类外部总权重/官方总权重固定为 **0.25**。仅比较原 E0c 五折与替换固定 E16 融合中 E0c 一路后的结果，另报逐折、逐类、按 prompt 配对 bootstrap 和来源敏感性。数据选样不使用官方验证标签。

**解释边界：** prvsiyan 的来源协议曾参考官方完整训练集的总体字数分布，并用带标签的家族长度现象决定按每次生成指定目标字数；这属于来源生成设计已利用官方训练信息。当前仅用其独立 `new` 原始行、排除全训练集衍生 funnel，显著减少直接泄漏，但五折结果仍应称为**探索性迁移诊断**，不能当作与官方数据完全独立的泛化证明。不同 API 默认推理设置和提示写法也可能影响与隐藏测试的匹配。公开代码须给来源署名；原始生成文本及 API envelope 不随代码发布，服务商输出使用条款还需按最终用途核对。
