# E3 外部数据获取与初审（2026-11-04）

本记录核对公开数据的访问、许可证、结构和基础来源信息。经过来源筛选的 Will Rice v5 数据已用于开发集 E3 探索；结果见 [results.md](results.md)。外部归档只保存在 Git 忽略的 `data/external/`。

## 获取情况

| Kaggle 数据集 | 官方 API 状态 | 本地状态 | 许可证 / 版本 |
| --- | --- | --- | --- |
| [Maniceet OpenRouter + OpenAI Generations](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations) | 匿名 `view` 和 `download` 均返回 HTTP 403：`Permission 'datasets.get' was denied`；作者公开数据列表里也没有它。同期同作者一个公开数据集的 `view` 返回 200，说明不是 Kaggle API 整体不可用。 | 无数据文件；无法核对行级字段或生成过滤清单。 | 搜索缓存中的数据卡称 CC BY 4.0、7,753 行、10 类；当前无法从源文件验证。 |
| [Will Rice OpenRouter Harvest](https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest) | 匿名 `view` 和 `download` 均返回 HTTP 200。 | 版本 5 的 ZIP 已保存于 `data/external/willrice/source_v5.zip`，并完成下述初审。 | Kaggle 元数据：CC BY 4.0、版本 5、`isPrivate=false`。 |
| [crtproduction Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) | 匿名 `view` 返回 HTTP 200。 | 只保存公开元数据，尚未下载约 117 MB 的大数据集。 | Kaggle 元数据：CC BY-SA 4.0、版本 14；描述写约 211,369 条，但仍称 v12，版本描述需再次核对。 |

Kaggle [官方公告](https://www.kaggle.com/product-announcements/485439)说明公开数据集可以通过 API 匿名下载；Maniceet 的 403 因而应视为该数据集当前的访问或可见性问题，不能假定换用登录凭据就能解决。若用户在 Kaggle 登录后能打开其数据页，可下载 ZIP 并放入本项目的 `data/external/maniceet/`；若页面仍显示无权限或不存在，就需改用可公开访问的来源。请勿发送 Kaggle 密码或 API token。

## Will Rice 版本 5 初审

- 归档 SHA-256：`f7fd25a04fb54a0a365906089047283286c6d9a7b7ebdc954dc826b4083f622a`。包含 `passages.csv`、`prompts.csv`、`calls.jsonl`、README 和三份试验记录；分析时直接读取 ZIP 内文件，未执行其中的内容。
- `passages.csv` 实有 **10,772 行、19 列**，其中包含 `label`、`requested_model`、`served_model`、`served_provider`、`setting`、`prompt_id`、`prompt`、`text` 等。`prompts.csv` 有 **1,000 个 prompt**。每条回答的 prompt ID 都能在提示表中找到，提示正文也逐行一致。
- 实际有 **13 个比赛标签**、14 个 `served_model` ID；README 所称“14 of the 18 models”与标签数不一致。OpenAI 五类各 1,000 条；`kimi-k3` 996；`glm-5.3` 975、Flash 978；Gemini 3.1 Pro 仅 20，3.6/3.7/3.8 Flash 分别 776/772/838；Qwen 417。五个 Claude 标签均缺失。
- 全部 10,772 行的 `requested_model` 与 `served_model` 相同，`served_model`、服务商、文本均非空，`finish_reason` 均为 `stop`。服务商分别为 OpenAI 5,000、Google AI Studio 2,406、Z.AI 1,953、Moonshot AI 996、Alibaba 417。
- Qwen 的 417 行由 `qwen/qwen3.8-max-prime` 357 行和 `qwen/qwen3.8-max-0902` 60 行组成，却共用比赛标签 `qwen3.8-max`。实际快照与比赛目标是否相符尚未得到主办方确认；**暂不将这 417 行视为已核实的精确标签**。数据作者还在 README 中指出 Gemini Flash 快照有不确定性。
- 行 ID 无重复，外部文本按大小写折叠及空白归一化后无内部重复。最初的纯精确重复初审读取过官方 `train.csv` 的 2,699 条文本，其中包括锁定保留集文本；没有发现重复。正式 E3 的精确及近重复筛查限定在 2,160 条开发集回答，未读取保留集；与任何开发回答的最高 char-4 TF-IDF 余弦相似度为 0.378，低于预定的 0.90 排除阈值。
- README 称提示正文来自 [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)，许可证 CC BY 4.0。该来源不是逐行字段；官方训练表也不含原始提示正文，因此目前只能核实外部文件内部的提示一致性，不能证明与官方提示在语义上完全无重叠。

正式 E3 仅纳入八个实际服务模型 ID 可核实的标签，每类按固定哈希选 250 条；过滤清单、近重复记录和五折结果见 `outputs/e3_willrice/`。两档预设训练权重均未优于只用官方数据的 E0c。外部提示与官方提示是否存在语义重叠仍需谨慎判断，尤其不应直接导入整个 Open Pool 或用外部数据对锁定保留集选模。
