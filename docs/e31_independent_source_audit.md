# E31：arxitekton 同提示生成集 v2 的独立来源审计

审计日期：2026-11-10。此轮只检查公开元数据与一个约 1.08 MB 的小型归档；未训练模型、未读取官方 30 个锁定保留 prompt，也未将第三方原文加入 Git。下文将**API 元数据和本地文件可直接核实的事实**、**发布者 README 声明**、**尚无证据的服务身份**分开表述。

## 固定来源和文件

- 数据集：[Name That LLM: same-prompt generations v2](https://www.kaggle.com/datasets/arxitekton/ntl-claude-gen-v2/version/1)，Kaggle ref `arxitekton/ntl-claude-gen-v2`、ID `12738399`。标题中的 **v2 是作者的数据集名称**；[Kaggle 公开元数据 API](https://www.kaggle.com/api/v1/datasets/view/arxitekton/ntl-claude-gen-v2) 于审计时返回的文件版本为 **1**，版本说明 `Initial release`，最后更新 `2026-10-20T18:03:06.677Z`，`isPrivate=false`。
- 元数据标示许可证为 **Attribution 4.0 International (CC BY 4.0)**；[固定 v1 的 README](https://www.kaggle.com/api/v1/datasets/download/arxitekton/ntl-claude-gen-v2/README.md?datasetVersionNumber=1)也这样自述，并提醒服务商输出条款可能另行适用。项目代码选用 MIT 不会将这份第三方数据改为 MIT；如再分发数据，须另行处理 [CC BY 4.0 的署名要求](https://creativecommons.org/licenses/by/4.0/)及适用条款。本项目不再分发此归档。
- 匿名下载固定 `datasetVersionNumber=1` 成功。归档保存在 Git 忽略的 `data/external/arxitekton/source_v1.zip`，**1,082,721 字节**，SHA-256 `076b439a1316a9c8ebfd8ce7169c19cd1cc3eaf8aa8c59341a682ecb10ccfd2f`。API 的 `totalBytes=3,060,828` 是下表三个文件的字节数之和，不是 ZIP 大小。

| 归档内文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `README.md` | 3,301 | `d14cc230bc0bc69cf124301a02d9301fbea40e6ecd578729af927c6ce41985e9` |
| `passages.csv` | 2,780,143 | `7861c8fd32fdee8f4e2d6be734adbe784e105b240a8b0aa53ddfc42d6cad7bb9` |
| `prompts.csv` | 277,384 | `f03030b21fcc18005501e41e18afd8bf731c0eddef17cec03a847618b11d2370` |

公开 API 的 `files` 字段为空，因此文件清单与列结构来自上述实际 ZIP，而不是从空的 API 文件列表推断。

## 实测 schema 与覆盖

`passages.csv` 实有 **3,322 行、11 列**：`id`, `prompt_id`, `model`, `text`, `n_words`, `n_target`, `setting`, `route`, `served_model`, `served_provider`, `s1_weight`。`prompts.csv` 实有 **995 行、5 列**：`prompt_id`, `domain`, `task`, `topic`, `prompt`。这两表的各列均无空值；回答 ID、`(prompt_id, model)` 组合、原始回答文本以及归一化空白和大小写后的回答文本均无内部重复。995 个提示 ID 唯一，回答的每个提示 ID 都能回连提示表。以上是**文件内一致性**，不验证生成过程或与比赛数据的重叠。

| 比赛目标标签（CSV `model`） | 行数 | CSV `route` |
| --- | ---: | --- |
| `claude-fable-5` | 936 | `openrouter` |
| `claude-opus-4-6` | 38 | `kaggle_proxy` |
| `claude-opus-4-8` | 42 | `kaggle_proxy` |
| `claude-opus-5` | 41 | `kaggle_proxy` |
| `claude-sonnet-5` | 46 | `kaggle_proxy` |
| `gemini-3.1-pro-preview` | 41 | `kaggle_proxy` |
| `gemini-3.8-flash` | 36 | `kaggle_proxy` |
| `gpt-5.5` | 36 | `kaggle_proxy` |
| `gpt-5.6-luna` | 57 | `kaggle_proxy` |
| `gpt-5.6-sol` | 45 | `kaggle_proxy` |
| `gpt-5.6-terra` | 48 | `kaggle_proxy` |
| `gpt-6-astra` | 37 | `kaggle_proxy` |
| `kimi-k3` | 962 | `openrouter` |
| `qwen3.8-max` | 957 | `openrouter` |
| **合计** | **3,322** | OpenRouter **2,855**；Kaggle proxy **467** |

覆盖比赛 18 类中的 **14 类**；缺 `gemini-3.6-flash`、`gemini-3.7-flash`、`glm-5.3`、`glm-5.3-flash`。`served_model`/`served_provider` 在每类中各只有一个 CSV 字段值：三类 OpenRouter 行分别写 `anthropic/claude-fable-5`/`Anthropic`、`moonshotai/kimi-k3`/`Moonshot AI`、`qwen/qwen3.8-max-0902`/`Alibaba`；其余 467 行的 `served_provider` 全为 `kaggle-proxy`，模型字段为代理 slug，例如 `anthropic/claude-opus-4-6@default` 和 `openai/gpt-5.5-2026-04-23`。这些字符串的**存在和计数已核实，实际后端服务模型未核实**。特别是 Qwen 只写 `-0902`，比赛标签未公布主办方生成快照，不能直接视为隐藏数据的精确版本证明。

README 称 `s1_weight` 每行均为 1.0，**与文件矛盾**：实际 3,162 行为 1.0，160 行为 0.5；后者正好是 `claude-sonnet-5` 46、`gemini-3.1-pro-preview` 41、`gemini-3.8-flash` 36、`gpt-6-astra` 37。不得直接使用发布者建议权重而不核对其形成方法。

README 的“same prompt”指共享提示库，而非完整的 995 × 14 矩阵：995 个提示中，只有 **23** 个由 14 类全部回答，**826** 个只有 3 类回答。`prompt_id` 可用于来源内部的提示分组，但这样的结构不能自动保证每类均衡比较。

## 逐行来源证据：不足

[v2 README](https://www.kaggle.com/api/v1/datasets/download/arxitekton/ntl-claude-gen-v2/README.md?datasetVersionNumber=1)声称 2026 年 10 月通过 OpenRouter 固定第一方服务商、关闭回退生成三类，其他 11 类走 Kaggle Benchmarks proxy；单条用户消息采用 `{prompt} Write a single paragraph of about {N} words, with no title or headings.`，`N` 在 40–250 间抽取，`setting` 记录推理档位。提示表确有全文，可按 `prompt_id` 关联，`n_target` 均为整数且在 40–250 之间。不过回答表没有逐行实际发送的 `prompt_sent`，无法核实模板确曾逐字提交。

更关键的是，ZIP **没有** `calls.jsonl`、原始 request/response envelope、`call_id`、响应 ID、生成时间、HTTP 状态、`finish_reason`、路由参数或代理后端身份记录。`id` 是 `g2:00000` 形式的行号，不能回连原始 API 调用。因而无法从原始响应验证任何一行的 `served_model`/`served_provider`、OpenRouter 是否真的仅走指定供应商且无回退、Kaggle proxy 的底层模型及快照、以及回答文本是否与响应一致。README 的“成功调用且 `finish_reason=stop`”是**发布者筛选声明**，不是文件中可复核的逐行状态。以当前发布文件计算，达到 E24 式“原请求、原响应、提示和服务身份逐行相互印证”门槛的行数为 **0**，不等于断言这些回答是伪造的。

## 与全部官方训练集的依赖

作者明确说 v2 延用 [v1 README 所述的 host-matched 协议](https://www.kaggle.com/api/v1/datasets/download/arxitekton/ntl-claude-gen-v1/README.md?datasetVersionNumber=1)：目标字数分布、体裁比例和提示主题来自官方训练集分析；每类生成设置由官方训练回答训练的风格判别器选择。v2 README 又明确列出下列过滤步骤：

1. 用比赛**训练及测试回答**做精确去重和共享 8-gram 排除；README 未披露它具体能访问多少测试回答，不能把“test”理解成全部隐藏测试集。
2. 用重构的 **150 个官方训练 prompt** 过滤主题，并删除接近训练提示组的外部提示组。
3. 按官方训练集中各类的标点/拼写特征剔除回答；仅保留由 `train-only judge` 评为足够“像官方训练集”的 `(label, setting)` 单元。

以上为**作者披露的筛选流程**；归档不含未过滤的原始调用和逐步筛除清单，无法独立重算每一步。尽管 README 说“不包含比赛数据”，这只说明发布文件自称没有复制比赛行，不能消除**发布前已依赖全部官方训练文本、标签及重构提示来设计和筛选**的问题。项目原有五折把 150 个官方训练 prompt 分成 120 个开发、30 个锁定保留；发布者的筛选已在划分前利用这 150 个 prompt 的信息。即使丢弃 `s1_weight`，保留的外部行也已受上述筛选影响，不能将“官方 + v2”在原 120 prompt 上的五折 OOF 解释为严格独立的外部数据增益。30 个锁定保留 prompt 也已被来源生成流程触及，不能靠本项目不读取保留集来撤销。

## 与其他独立来源的比较及决定

这份数据由不同于 prvsiyan 的作者发布，也有可回连的 995 条提示正文；但**可审计的服务身份明显弱于** [prvsiyan v9 `new` 原始来源](e24_prvsiyan_source_audit.md)：后者有 13,213 条原始 API envelope，可将 5,400 条 `new` 回答逐行与请求、响应、提示、路由和停止状态核对。prvsiyan 自身仍有全官方训练长度统计依赖，其五折也只作探索性诊断。

当前另一个更值得追溯的独立作者来源是 [Will Rice OpenRouter Harvest v5](e3_public_data_access.md)：本地包有 `calls.jsonl`、`prompts.csv` 和请求/返回模型、服务商字段，可从原始调用重建候选；但其已清洗的 10,772 行同样受全部官方训练集的标点规则和训练集风格判别器影响，**不能直接作为严格五折外部训练集**。[Maniceet 来源](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations)在本次匿名元数据复查仍返回 HTTP 403，暂无可审计文件。没有发现已可直接准入、又完全独立于 prvsiyan 且满足严格五折门槛的小型来源。

**决定：E31 对 arxitekton v2 判为 no-go，当前准入训练行数 0；不运行五折，也不把其 `served_model` 当已确认的目标版本。** 若仅用于明确标注来源偏移的描述性诊断，应保留版本和哈希，并避免将结果写成独立 OOF 或隐藏测试性能证明。要重审逐行身份，至少需发布者提供原始请求/响应、真实后端/服务商及回退参数、逐字提交提示、时间与停止状态；即使补齐调用证据，当前全训练集驱动的提示设计和筛选仍限制严格五折结论，需要一个未参考验证折重新生成和筛选的来源。
