# 外部训练数据候选（E3 前核对）

比赛允许免费公开的外部数据与预训练模型，但模型来源分类需要**具体版本**标签。不能仅凭 API 请求时的模型别名推断实际生成版本；服务商路由与回退记录也需核对。[比赛说明](https://www.kaggle.com/competitions/name-that-llm/overview/description) · [OpenRouter 路由说明](https://openrouter.ai/blog/insights/model-routing/)

| 候选 | 当前可核实资料 | 初步处理 |
| --- | --- | --- |
| [Maniceet: OpenRouter + OpenAI Generations](https://www.kaggle.com/maniceet/name-that-llm-openrouter-openai-generations) | 数据卡称 7,753 条、10 个比赛标签、CC BY 4.0；逐行给出路由、服务商、模型 ID 和 prompt 来源，GLM 两类占多数 | 首选小规模 E3；先筛掉实际版本与服务商无法确认、prompt 来源不完整的行 |
| [Will Rice: OpenRouter Harvest](https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest) | [公开元数据镜像](https://silytics.si/datasets/kaggle%3Awillrice/name-that-llm-openrouter-harvest)称约 10,772 条、14 个标签、CC BY 4.0；原始数据卡目前未核对 | 检查逐行 served model、服务商与类别数量后再纳入 |
| [crtproduction: Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) | [数据简介](https://papersdatacode.com/datasets/43ef5b24-name-that-llm-open-pool-255k-passages-18-llms)称约 254,561 条、18 类、59 来源、CC BY-SA 4.0 | 暂不整体导入；逐来源核对标签映射、许可证和泄漏标记后按来源消融 |

导入流程：保存 URL、版本、下载日期、许可证与行级 provenance；确认实际服务模型的完整 ID；在开发集范围内过滤与官方文本的精确/近重复，检查 prompt 重叠；仅在相同开发折上比较官方数据与“官方 + 外部数据”。最终保留集不参与筛查、训练或选模。外部数据的类别不平衡与生成提示差异需在训练时控制。公开 GitHub 仓库优先发布下载与清洗代码及署名清单，不直接转存未审计的第三方数据；CC BY-SA 再发布的改编数据须遵守[相同许可要求](https://creativecommons.org/licenses/by-sa/4.0/)。
