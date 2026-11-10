# E12 外部数据来源审计：暂不进入训练

审计日期：2026-11-05。此轮没有运行五折消融，也没有重新使用 30 个保留 prompt。`run_e12_external_audit.py` 只读取一个小型原始来源的 CSV，不读取官方训练集或保留集。

## 汇总池当前状态

[crtproduction 的 Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) 在本次核对时为 **Kaggle 文件版本 15、数据卡内部版本 v13**，含 **212,623** 条主文件回答、18 个比赛标签、60 个来源，页面许可证 **CC BY-SA 4.0**。这比本地 2026-11-04 保存的 Kaggle 版本 14／内部 v12（211,369 条、59 个来源）更新；v13 的 1,254 条新增回答全部经 Kaggle Benchmarks proxy。数据卡称主文件移除了 284 条回答官方训练 prompt 的文本，以及 10,838 条无可信 Vertex AI 路由的 Gemini 3.7 文本。这些是发布者的筛查声明，并非我们逐行复核的结论。

主文件说明中的 `model` 是**原始发布者宣称的标签**。主文件有 `source_dataset`、`source_file`、`source_license`、`origin_hint`、`prompt_canon`、`prompt_evidence`，但**没有逐行 `served_model`、请求路由或实际服务商字段**。数据卡也明确说没有重新验证每条回答来自哪个模型。因此，不能把 212,623 条都当作精确版本已核实的样本。`prompt_evidence` 中仍有 40,261 条只依据 ID、1,925 条无法恢复 prompt、14,684 条通过内容质心推断；后者的匹配有发布者估计的误差。单看 `prompt_canon` 不足以验证全部提示来源。

数据卡的 `match_weight` 以**全部 150 个官方训练 prompt** 的排版和长度统计为目标拟合，`style_score`、`max_cos_train` 等也使用官方训练集。如果用这些列筛选或加权后，在原 120 个开发 prompt 上报告五折 OOF，验证折的信息已影响外部样本选择。即使不使用这些列，仍需从每个原始来源文件逐行回连到实际服务模型、服务商及提示文本，不能直接以汇总池标签代替核验。

## 小型原始来源交叉检查

为检查是否存在可审计的替代入口，下载了 [zeminglii 的 NTL own exact-route generations](https://www.kaggle.com/datasets/zeminglii/ntl-own-exact-route-generations) **Kaggle 版本 2** 的 `passages.csv`（约 1.81 MB）。本地 SHA-256 为 `4ff8c42f3c26b8ee0cfd1552253c3b7faa2ad4d111a8f8781a292e8d41e62afe`。发布者数据卡写明通过 Kaggle proxy 或限定第一方供应商且关闭回退的 OpenRouter 生成，许可证为 **CC BY 4.0**；CSV 中有 `model`、`route`、`provider`、`model_seen`、`finish_reason`、`prompt_canon` 等字段。

独立脚本 `python run_e12_external_audit.py` 实测 **2,162 行、17 个比赛标签**，全部 `finish_reason=stop`，`model_seen`、`prompt_canon` 均非空，精确归一化回答无重复。但 **`qwen3.8-max` 没有样本**；**1,291 条 Kaggle proxy 回答的 `provider` 为空**；文件只有提示 ID，没有提示正文。OpenRouter 行的服务商字段有 Z.AI 532、Google 319、Anthropic 10、Moonshot AI 10 条。这里的路由和模型 ID 仍是来源文件的记录，不能证明底层服务端没有静默版本切换。审计结果保存在 Git 忽略的 `outputs/e12_external_audit/zeminglii_v2.json`。

已做的 [Will Rice v5 E3 实验](results.md) 只纳入八个版本可核实类别，每类 250 条。预设的 0.25 外部训练质量比得到 0.3421 OOF，低于官方 E0c 的 0.3426；1.0 比例得到 0.3292。该小型来源不能填补汇总池中 Qwen 等来源审计缺口，也不能用 E3 的结果推断 18 类汇总池的收益。

## 判定与解锁条件

**No-go：当前不下载整份 Open Pool 主文件，也不运行 E0c＋外部数据的五折训练，更不将外部原文或衍生样本上传至 GitHub/Kaggle。** 需要先取得至少一个覆盖缺失 Qwen 类的、逐行带实际服务模型 ID、服务商／路由、提示正文或可审计提示索引的原始来源；其余类别也要按原始来源核对，而非仅信任汇总池的 `model`。各来源版本和许可证应锁定，且要排除官方训练 prompt 衍生回答。

解锁后沿用原 120 个开发 prompt 和五折 GroupKFold，只用开发文本做去重和近重复筛查，不读取 30 prompt 保留集；按来源及类别限额采样，预设低外部质量比 0.25，并以相同折的官方 E0c 为对照。不能用以全部官方训练集拟合的 `match_weight` 或 `style_score` 为开发 OOF 选模。第三方原文继续留在 `data/external/`（Git 已忽略）；开源代码应附来源、版本、许可证及署名，若再分发改编数据需遵守各来源许可及汇总池的 CC BY-SA 条款。
