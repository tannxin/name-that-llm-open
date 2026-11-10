# E29：Qwen 0902 快照的固定敏感性实验协议

预登记时间：2026-11-10，**审计和五折拟合之前**。E24 的已提交 Notebook 与模型工件不修改。本实验只问一个问题：在原 E24 17 类外部样本、折号、E0c 特征/超参和 E16 固定融合完全不变时，加入同一 prvsiyan v9 原始 `new` split 的 Qwen 0902 回答是否有可复现的增益。

## 背景和解释范围

[阿里云第一方模型文档](https://help.aliyun.com/en/model-studio/qwen3-8-max)将服务别名 `qwen3.8-max` 映射到快照 `qwen3.8-max-0902`，使 prvsiyan 原始 API 返回的 `qwen/qwen3.8-max-0902` 成为合理候选。但 [比赛](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)没有公开目标生成时的实际服务快照、路由和解码参数。这只支持**版本敏感性探索**，不能把候选与隐藏标签等同。该来源的生成协议参考过完整官方训练集长度，后续 OOF 仍是探索性结果。

## 审计门槛：先审计，失败即停止

锁定 prvsiyan v9 的 `passages.csv` SHA-256 `393f300c7b521507a55f5a8793bd2256519ed6324da97153f8cd2b3503f01ce0`、`prompts.csv` SHA-256 `d1424468ca172e3e589a996c8a2f9a0ee5339dc3443104ba05e27bd1cde3be48`、`provenance.jsonl` SHA-256 `a7e78ff29ba2510fc9180cfcf864e0a1ecc1c36ee1bbd059785b2ad95b2a94de`。核实 Kaggle 版本 9、公开、CC BY 4.0。只取 `split=new` 且标签 `qwen3.8-max` 的**全部 300 行**，要求恰好 300 个不同提示、每提示一行、非空/正常 stop/未标拒答或截断、行 ID 和 `call_id` 唯一。逐行以 `call_id` 匹配原始 envelope，核对 CSV/原请求/原响应的提示、模型、服务商、回答、响应 ID、完成状态，HTTP 200 和 identity `ok`；原请求路由 OpenRouter，模型固定为 `qwen/qwen3.8-max-0902`，只允许 Alibaba 官方服务商且 `order=only`、`allow_fallbacks=false`，原始返回的 `model` 必须精确等于该 ID。每个 `prompt_sent` 必须与 `prompts.csv` 的 `new` 模板和目标字数逐字匹配，提示许可 CC BY 4.0，`source_prompt_id` 为空。

只用原 E0 的 **120 个开发 prompt、2,160 条回答文本**，以与 E24 相同的 char 4-gram TF-IDF（`min_df=2`、`max_features=100000`、lowercase）测每条 Qwen 回答对官方开发集的最高余弦，排除 `>=0.90` 的行，不回填；报告 `>=0.5`、`>=0.8`、`>=0.9` 数量和最大值。不得触及旧的 30 prompt 锁定保留集，不用 `passages_clean.csv` 或来源 funnel/相似度/风格筛选列，也不用官方验证标签选样。若 SHA、许可、路由、快照、prompt 或 envelope 任一条件失败，**不运行拟合**。

## 唯一拟合对照与晋级门槛

复用 E24 已锁定的 250 个外部 `prompt_id`（原规则：`SHA256('name-that-llm:e24:v1:seed42:' + prompt_id)` 排序前 250），仅从审计合格的 300 条 Qwen 行中取对应提示，不回填；预期最多 250 条。E24 原 **3,620 条**17 类外部数据逐行不动。每折官方仍有 96 行/类，官方权重 1；Qwen 的审计合格选中行每行权重 `24/n_qwen`，使新增类的外部总权重恰为官方该类的 0.25；原 17 类各自总外部权重 24 不变。沿用原 E0c 词 1–2 gram、字符 3–5 gram、57 个 style 特征、`LinearSVC(C=1, class_weight='balanced', dual=True, random_state=42)`，同一 120 prompt/2,160 行和五折分组，所有变换器只在各折训练行拟合。

先重算并逐行复现 E24 的 E0c 决策分数与已保存文件；若复现失败则停止。只改变 Qwen 行的加入，得到新的 E0c OOF。比较：E24 E0c 单路 vs E29 E0c；固定 E16 融合中只把 E24 E0c 分数换成 E29 E0c，RoBERTa、三个 ModernBERT seed、逐行 z 标准化及三路等权不变。逐折、逐类报告 balanced accuracy/recall，120 prompt 配对 bootstrap **20,000 次、seed 42** 95% 区间，错转对/对转错及新来源行数/哈希。**晋级门槛相对于 E24 固定融合**：总体 BA 至少 `+0.01`、至少 4/5 折不降、任何类别召回不得下降 `>=0.05`。任一不达即停止，不按 OOF 或公榜调抽样、权重或融合，也不做全量模型及 Kaggle 提交。若通过，只作为后续独立复核候选；不把同一来源的五折当作外部泛化证明。
