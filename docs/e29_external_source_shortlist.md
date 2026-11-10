# E29：可独立审计的外部回答来源候选

日期：2026-11-10。**只调查来源，没有下载新语料、训练或改动 E24 准入清单。** 优先级按“能回查实际服务身份、提示、许可证，以及与当前 E24 来源独立”排列。候选行数是发布者卡片或已下载原始包中的数量，**不是已准入训练的数量**。新来源须先固定版本和文件 SHA，再核对每行 `prompt_id → 实际提示 → 请求模型/服务商 → 返回模型/服务商 → 回答/完成状态`，最后查与 E24 及官方开发回答的重复。官方五折只按既定 prompt 分组评估，不用外层验证标签筛源。

## Qwen 版本映射的新证据

[阿里云 Model Studio 的 `qwen3.8-max` 官方模型页](https://help.aliyun.com/en/model-studio/qwen3-8-max) 明确列出模型 ID `qwen3.8-max` 的快照为 **`qwen3.8-max-0902`**，别名 `qwen3.8-max-2026-09-02`；[OpenRouter 的 `qwen/qwen3.8-max-0902` 页面](https://openrouter.ai/qwen/qwen3.8-max-0902)也将其标为该快照，而 [`qwen/qwen3.8-max-prime`](https://openrouter.ai/qwen/qwen3.8-max-prime) 是单独 SKU。[比赛标签表](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)仅给 `qwen3.8-max`，没有主办方生成时的时间、服务商、快照或解码参数。**因此别名映射已有第一方证据；这仍不能证明比赛隐藏回答实际由 0902 和相同服务配置生成。** 当前 E24 排除 Qwen 的固定协议保持原样；可以另立一次预登记敏感性分支，只测试已审计的 prvsiyan `new` 里 300 条 `qwen/qwen3.8-max-0902`，与 E24 以同一折/相同融合规则比较，不混入 `-prime`。

如果未来自行生成补充数据，[阿里云 OpenAI 兼容 Chat API 文档](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)允许明确请求固定 `qwen3.8-max-0902`；必须记录地区、请求/返回模型、完整提示、thinking 设置、停止原因、生成时间和许可。自行生成会发生 API 成本；本报告没有发起调用，也没有把未公开的新文本当作比赛合格公共数据。

## 候选清单

| 来源 | 覆盖与可审计字段 | 可能纳入的行与当前障碍 | 下一步 |
| --- | --- | --- | --- |
| [prvsiyan v9 原始集](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/version/9) | 本地已逐条核验 `new` 5,400 行/18 标签，对应 13,213 条原始 API envelope；有完整提示、`call_id`、路由、请求与返回模型、返回服务商和停止状态。E24 已准入 17 类 4,322 行，哈希抽样 3,620 行。[审计](e24_prvsiyan_source_audit.md) | **300 条 Qwen `-0902`** 是最快可测的第 18 类候选；仍是同一来源，不能替代独立来源检验。原生成协议曾参考完整官方训练集长度/家族现象。 | 单独冻结“E24+Qwen”消融的选样与权重；先比较来源内/官方开发类内风格差异，再做五折和来源外验证。 |
| [arxitekton same-prompt generations v2](https://www.kaggle.com/arxitekton/ntl-claude-gen-v2/discussion) | 公开数据卡简介称 **3,322 条、14 个候选模型**，扩展 v1，使用 host-protocol；当前可读页面未公开逐行 schema、服务快照和确切许可。 | **尚不能指定可准入行。** 若原文件含 API 响应或可追溯调用 ID，这是优先审计的独立候选；缺少第 18 类的证据。 | 先只查版本卡、文件名/小型 README 与列名，核实 14 个具体标签、原始调用、提示来源和许可证；不直接导入。 |
| [Maniceet OpenRouter + OpenAI Generations](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations) | 数据卡称 **7,753 行/10 标签，CC BY 4.0**；每行有 `api_route`、`provider`、`served_model`、`record_id`、`prompt_exact`。其 `zaiglm` 为固定 Z.AI 且禁回退；`oa56`、`oa56b` 为 OpenAI API；`orglm` 则走最便宜 OpenRouter 服务商。 | 可优先审计 `zaiglm` **768 行**和 `oa56`/`oa56b` **1,466 行**的原始记录。卡片说 `served_model` 在未保留原响应时为空，`own` 的提示并非逐字请求；全部提示还曾对**完整官方训练回答**筛相似度，因此发布文件不能直接当严格五折独立来源。Qwen 仅 5 行。此前匿名 API 访问曾返回 403。[旧访问记录](e3_public_data_access.md) | 先重新确认文件可公开访问，再按 `record_id` 回查原响应与未筛选的 prompt；若只拿到已由全官方训练数据筛过的文件，限定为探索性来源外诊断。 |
| [Will Rice OpenRouter Harvest v5](https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest) | 本地 ZIP 含 `calls.jsonl`、`passages.csv`、1,000 个提示及设置试验；清洗表为 **10,772 行/13 标签**，有请求/返回模型、服务商、提示、停止状态，CC BY 4.0。[本地初审](e3_public_data_access.md) | 原 `passages.csv` 的 **60 条 Qwen `-0902`** 可能作版本敏感性资料；**357 条 `-prime` 不合并**。README 明说用完整比赛训练集的 Gemini 长破折号现象过滤回答，并用全训练集训练的风格分类器选择生成设置；故清洗表不适合严格官方五折。E3 的旧小样本亦未带来收益。 | 如需用它检验 E24 来源迁移，应从 `calls.jsonl` 重建未经过全训练标签驱动清洗的固定规则子集，并逐行核验提示；仅作为探索性对照，不把旧清洗表扩大训练。 |
| [zeminglii exact-route generations v2](https://www.kaggle.com/datasets/zeminglii/ntl-own-exact-route-generations) | 本地 `passages_v2.csv` **2,162 行/17 类，CC BY 4.0**；有路由、`provider`、`model_seen`、完成状态，OpenRouter 固定服务商部分 **871 行**（Z.AI 532、Google 319、Anthropic 10、Moonshot 10）。[本地审计](e12_external_data.md) | `prompt_canon` 是提示 ID，而文件没有提示正文；另 **1,291 条 Kaggle proxy** 无实际服务商。Qwen 缺失。严格准入行目前为 **0**。 | 找到原始提示表及逐字请求/返回 envelope，先审计 871 行的提示和去重；代理行继续隔离。 |
| [Tadeu Hayashida Claude Fable 5](https://www.kaggle.com/datasets/tadeuhayashida/ntl-claude-fable-5-generations) | 本地保存的 Kaggle 搜索元数据称约 **1.57 MB、CC BY 4.0**，标题指向 `claude-fable-5`；未获得可读行级说明。 | 能否补弱类未知；**无行可直接准入**。 | 只取元数据/README 和列名，要求第一方 Anthropic 返回模型、原提示、许可与完整响应，再决定是否下载小文件。 |
| [Zakir Khan Benchmarks proxy generations](https://www.kaggle.com/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations) | 数据卡称 **834 条 GPT/Gemini、CC BY 4.0**，有逐字请求、目标字数及 proxy slug；提示来自独立公共提示库，作者称不含官方回答。 | `proxy slug` 不能单独证实实际后端服务商/快照；严格精确标签暂 **0 行**。提示目标字数来自官方训练直方图。 | 可作不依赖版本身份的提示/格式稳健性诊断；若取得代理完整原始响应与后端证明，才考虑精确标签。 |

### 暂不作为新数据直接训练

[Open Pool v15](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) 汇总主文件虽覆盖 18 标签、规模大，但其公开卡片说明曾用全部 2,699 条官方训练回答及全部 150 个提示评估、筛选来源/路由，并有依赖全训练集的衍生列；主文件又缺逐行真实服务模型/路由。因此删除 `match_weight` 等列仍无法还原无验证折影响的原始来源。[原审计](e24_prvsiyan_source_audit.md) 同理，其他合并后的“18 类大池”只能作**原始来源索引**，不能因标签数齐全就直接拟合。

## 执行次序和验证边界

1. 先做**不需 GPU**的元数据与小文件审计：prvsiyan Qwen `-0902` 的固定敏感性实验设计；arxitekton 的 schema/许可/原调用可追溯性；Maniceet 的可访问性及 `record_id` 回查。任何来源都先排除与 E24 的重复 prompt/回答，不能把重复数据当独立验证。
2. 新来源只有在逐行身份、服务商、提示、许可、来源时间和全训练集筛源风险全部说明后，才进入预先固定的 120 prompt、五折 GroupKFold。跨来源验证应单独报告，尤其针对 [E26 发现的 Gemini 标点漂移](e26_domain_audit.md)：E24 Gemini 3.7 外部长破折号出现率 28.4%，官方开发仅 0.8%。
3. 真实隐藏集仍不可读，本地测试文件只有三个交互预览样本。官方开发 OOF、来源外测试和 Kaggle 公榜各回答不同问题；不能根据一次公榜倒推逐类阈值或来源过滤。
