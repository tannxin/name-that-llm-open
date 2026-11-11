# P1-3：Oleg 公开回答语料的来源预审（2026-11-11）

本轮只读审计 [Name That LLM Response Corpus](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus) 的公开数据卡、文件清单及五个小型元数据文件；未下载回答 Parquet、未训练模型、未接触锁定保留集。下列计数是发布包的自检结果，尚未通过本项目逐行重算。来源能否进入严格五折，仍以 [P1-2 固定准入协议](p1_2_external_source_refresh_2026-11-11.md)为准。

## 固定版本与文件

- [Kaggle 元数据 API](https://www.kaggle.com/api/v1/datasets/view/olegzholobov/name-that-llm-response-corpus) 标识公开数据集 `olegzholobov/name-that-llm-response-corpus`，文件版本 **1**，快照日期 **2026-10-15**。许可证栏为“Other”；[README](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus)明确内容来自不同许可证，发布包未给上游内容重新授权。
- [文件清单 API](https://www.kaggle.com/api/v1/datasets/list/olegzholobov/name-that-llm-response-corpus) 返回 15 个文件。通过 Kaggle 的 `datasets.DatasetApiService/DownloadDataset` 单文件接口只读取 `README.txt`（6,198 B）、`data_dictionary.json`（3,182 B）、`manifest.json`（2,428 B）、`source_catalog.csv`（7,151 B）、`validation.json`（5,619 B）。
- `manifest.json` 给出的 SHA-256：`source_catalog.csv` 为 `c1033245fa80f8c0ca01104c8aeb2a76be0b823aec111f0c77d04345faec53d3`；`validation.json` 为 `fd0fbb3307a12fe47852e84086275f74a82744776bcc5912f0c219b1820da17e`；`external_cell_provenance.parquet`（40,625,399 B）为 `906345a4d1f418da63267bf686e1e6847c09f347ecf9eb939860d9aae32abdd6`。前两个小文件已独立重算 SHA 并吻合；此轮没有下载后者，故未重算其哈希或行级字段。

## 发布包声称的覆盖与限制

| 视图 | 宽表行 | 非空答案格 | 唯一回答 | 18 类完整提示组 |
| --- | ---: | ---: | ---: | ---: |
| 全量 `external_all_wide.parquet` | 70,980 | 164,954 | 135,171 | 1,902 |
| 40–250 词 `external_short_wide.parquet` | 13,642 | 24,025 | 24,015 | **0** |

两视图重叠，不可相加为独立样本。短集是机械筛选结果，不等于许可或完成状态认证；其 18 类覆盖很不均衡，例如 Gemini 3.6/3.7/3.8 Flash 分别只有 83/77/71 格。全量 `validation.json` 还报告 49,625 条 canonical 回答缺停止元数据，69,089 条需审查输出使用条款，22,291 条来源许可证未知；这些计数可重叠，也不能直接套用于短集。单个回答可能关联多个来源出现次数，所以 188,470 次 provenance occurrence 不是独立训练样本数。

README 称对 2,699 条官方训练答案及三条可见测试答案没有发现字节、规范化或长文本包含的重复；其模糊匹配检索有共享 trigram 前提，不能排除所有改写。官方训练文件没有提示正文，故回答排重不能证明外部提示与每个官方验证组独立。发布包只用逐字相同且经核实的 user prompt 分组；相同 user prompt 也可能有不同 system prompt、历史、图像和采样设置。逐行训练前须按 `prompt_group_id` 分折，并同时隔离重复 `response_id` 和相关轨迹。

## 身份证据层级与候选来源

`source_catalog.csv` 和 `validation.json` 使用的是**来源发布者的证据类别**，不是本项目对原始 API 调用的独立认证。较强的 `request_and_returned_model` 共 53,838 条 canonical 回答，恰好对应下表前两源的唯一回答数之和：

| 来源 | 来源出现次数 / 唯一回答 | 比赛目标类数 | 目录所列身份凭据 | 许可与现阶段判断 |
| --- | ---: | ---: | --- | --- |
| [Speechmap 原始仓库示例](https://github.com/xlr8harder/speechmap-data/blob/476ea8f0adf85b737bbb6e38e838448a8d4f2a9b/responses/us_hard_anthropic_claude-fable-5.jsonl) | 52,640 / 52,640 | 18 | `request_and_returned_model` | 仓库声明 Apache-2.0；抽样原始行有请求模型、返回模型、上游服务商、响应 ID、完成状态，但缺逐次原始请求正文和历史路由/回退设置；严格准入 **0 行**。 |
| [Woog arena-prose](https://huggingface.co/datasets/woog/arena-prose-100-49-models/tree/66298b561a69c6ed6a9a3c4c79eeedb1f754ba49) | 1,198 / 1,198 | 12 | `request_and_returned_model` | 提示 CC BY 4.0；[固定版本 README](https://huggingface.co/datasets/woog/arena-prose-100-49-models/blob/66298b561a69c6ed6a9a3c4c79eeedb1f754ba49/README.md)只列请求/返回模型与请求/响应哈希，没有原始 envelope；[LICENSE](https://huggingface.co/datasets/woog/arena-prose-100-49-models/blob/66298b561a69c6ed6a9a3c4c79eeedb1f754ba49/LICENSE.md)不授予生成输出的概括使用权。当前严格准入 **0 行**。 |
| [Eventual VLADBench](https://huggingface.co/datasets/Eventual-Inc/VLADBench-reeval) | 44,772 / 3,658 | 4 | `row_request_response_model_id` | 回答增补 CC BY 4.0；上游问题未标明许可、帧另依原条款。重复 occurrence 多，且多模态任务与本赛短文本差异大。 |
| [Flavourbench](https://huggingface.co/datasets/josefchen/flavourbench) | 4,272 / 764 | 8 | `row_request_returned_model_source_reported` | 提示/元数据 CC BY 4.0；回答保留服务商条款。需核对原始返回模型与完整上下文。 |
| [Pamela Zenodo 21278557](https://zenodo.org/records/21278557) | 683 / 532 | 4 | `row_request_returned_model_source_reported` | 目录列 CC BY 4.0；需核对每条原始调用及输出适用条款。 |

其他来源存在 `raw_api_response_model`、请求模型、运行元数据、基准文件标签或作者自述等不同层级；这些标签不能互相替代。README 也明确说模型身份依据来源发布的标签与元数据，**并未逐条独立认证每个 API 响应**。同名比赛类别不能自动证明其具体服务快照相同。

## 决定与下一道门槛

**当前严格准入训练行数为 0（尚未完成原始行级审计），不是断言语料不真实。** 全量的 1,902 个 18 类完整组不能当作短文本匹配组：短集此类组为 0，且即使 user prompt 相同，完整请求上下文也可能不同。不能凭目录计数立刻做 E0c/GPU 训练，更不能依据这份语料直接推断 Kaggle 增益。

Speechmap 的[模型目录](https://github.com/xlr8harder/speechmap-data/blob/476ea8f0adf85b737bbb6e38e838448a8d4f2a9b/model_catalog.jsonl)及[采集脚本](https://github.com/xlr8harder/speechmap-eval/blob/main/ask.py)只给出默认配置，不能证明每条历史调用实际使用的系统提示、上下文和 OpenRouter 回退设置。抽样返回模型虽与部分目标版本同名，比赛隐藏测试的服务快照并未公开；争议话题与拒答也有明显领域差异。

下一步若能取得 Speechmap 逐次原始请求和路由日志，再逐条建立 `source occurrence → 原请求/逐字提示/上下文与路由参数 → 原响应/实际返回模型及服务商/停止状态 → 发布答案` 映射；若 Woog 发布原始调用及对应输出权利依据，也按同一标准重审。还须核查来源生成和筛选未使用官方 120 个开发与 30 个锁定保留 prompt 的回答或标签，并逐项确认许可。随后在**不看五折分数前**冻结可准入行清单、去重、类别权重、文件哈希和来源域检查。只有上述证据齐全，才按 [P1-2 预设 E0c 筛查](p1_2_external_source_refresh_2026-11-11.md)决定是否进入 GPU 实验。
