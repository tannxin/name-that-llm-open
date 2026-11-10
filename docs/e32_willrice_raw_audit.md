# E32：Will Rice v5 原始调用的独立来源审计

审计日期：2026-11-10。只读取 Git 忽略的本地归档 `data/external/willrice/source_v5.zip`，不训练、不读取官方锁定的 30 个 prompt，也不发布第三方回答原文。准入标准沿用 [E24 行级审计](e24_prvsiyan_source_audit.md)：原请求、原响应、提示、实际服务模型与供应商、完成状态须能逐行相互核对；同时不能把利用全部官方训练集筛选过的外部行当作严格独立的五折证据。

## 归档与文件内事实

- 公开来源为 [Will Rice OpenRouter Harvest](https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest)。本地 Kaggle 元数据记录文件版本 **5**、公开、CC BY 4.0；归档 SHA-256 为 `f7fd25a04fb54a0a365906089047283286c6d9a7b7ebdc954dc826b4083f622a`。归档 README 的“Versions”只写作者内容版本 1 和 2，不能用它代替 Kaggle 文件版本号。
- `calls.jsonl` 实有 **12,308** 行、全部为同一组 **14 个扁平字段**：`model_id`, `served_model`, `served_provider`, `finish_reason`, `text`, `n_words`, token/cost 统计, `prompt_id`, `model`, `target_words`, `setting`。13 个比赛标签、14 个 `model_id`（Qwen 分成两个快照）；所有行的 `model_id` 等于 `served_model`，服务商字段非空，`prompt_id` 均在 `prompts.csv` 的 1,000 个唯一 ID 中。`n_words` 均等于该行文本的空白分词数。
- 原调用表有 **12,277 行 `finish_reason=stop` 且文本非空**，其余 31 行为 `length`，其中 24 行文本为空；按归一化文本计的 23 个重复来自空文本。`passages.csv` 的 **10,772** 条清洗回答都能通过模型、服务商、设置、prompt ID、目标字数及文本唯一对应回 `calls.jsonl`；清洗表的提示正文与 `prompts.csv` 完全一致。这证明发布文件彼此一致，不能验证真实 API 请求和返回。
- `pilot.jsonl` 300 行、`trials.jsonl` 520 行、`setting_trials.jsonl` 1,210 行同样只有上述扁平字段；分别有 240、320、160 行与主 `calls.jsonl` 精确重合。它们没有提供额外的原始 API envelope。未把试验记录与主调用合并，以免重复计算或混入根据官方风格选出的设置。

## 筛选依赖与行级身份边界

归档 README 明说 `calls.jsonl` 包括后来被清洗表剔除的主运行调用。因此，从它重建固定规则子集可避开**清洗阶段**的 Gemini 长破折号排除、与官方训练/测试文本的完全重复排除，以及按官方字数范围过滤。可是，**生成阶段**已用完整官方训练回答的字数分布抽取 `target_words`；作者又用完整官方 `train.csv` 训练风格分类器，以其 fidelity 选择模型推理设置。即使只取 `calls.jsonl` 而忽略 `passages.csv`，也无法消除这些先于主运行的设计依赖。初始 pilot 可能先于设置选择，但仍有相同的来源字段缺口，不能据此宣称严格独立。

更直接的限制是：四个 JSONL 文件都没有实际发送的消息正文、系统/推理与路由参数、HTTP 状态、`call_id`、响应 ID、原始响应体或独立的后端日志。`model_id`、`served_model`、`served_provider` 和 `finish_reason` 只在作者整理的同一行中出现；前两者全部相等是内部一致性，**不是**从原始响应独立核实实际服务模型、原厂路由或 fallback 关闭。`calls.jsonl` 只有 `prompt_id` 和目标字数，无法核实 README 所述模板是否逐字发送。Qwen 的 360 条 `-prime` 与 60 条 `-0902` 仍不能合并为一个已证实的比赛目标快照；Gemini Flash 实际快照也未获主办方确认。

**结论：达到 E24 式“未受全官方训练集选择影响，且原请求/响应/服务身份可逐行核实”的联合准入行数为 0；E32 不训练、不提交。** 这不表示 12,277 条非空正常停止回答无效，只表示发布材料不足以给出严格五折增益或精确后端身份结论。如将来获得原始请求/响应及生成前的固定协议，再按官方外层训练折重新设计或重生成长度/设置，才可重审独立性；在此之前，`calls.jsonl` 最多用于明确标注来源偏移的探索性诊断。
