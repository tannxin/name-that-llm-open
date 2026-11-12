# P1-5：公开外部来源增量复查

2026-11-12，仅核对 Kaggle 公开数据卡、版本、文件清单与来源说明；不下载回答主表、不使用保留集或 GPU。准入仍按 [P1-4](p1_4_pool_v16_source_audit.md) 的逐行原始请求/响应、实际服务模型与版本、独立提示、完成状态和许可门槛。

| 来源 | 本轮变化与核对 | 新增严格准入 |
| --- | --- | ---: |
| [kilupy Claude passages](https://www.kaggle.com/datasets/kilupy/ntl-claude-family-passages) | 仍为文件 v2；[元数据](https://www.kaggle.com/api/v1/datasets/view/kilupy/ntl-claude-family-passages)的许可证为 Unknown，文件清单仍无可逐行回连的原始 API 调用记录。 | 0 |
| [prvsiyan extended dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) | 更新至 v13，新增 `train_regen_w1`：1,726 条回答来自 135 个重构的比赛训练提示，且按官方训练回答做主办方风格相似性筛选；已审计的独立 `new` 分支没有因此新增行。其[伴生数据集](https://www.kaggle.com/datasets/prvsiyan/the-paragraph-without-a-signature)也未提供原始调用 envelope。 | 0 |
| [aitahmednasser Kaggle AI generations](https://www.kaggle.com/datasets/aitahmednasser/ntl-kaggle-ai-gen) | 新列表只有两个代理生成回答 CSV，未见 README、原始 envelope 或实际服务后端证据。 | 0 |
| [Open Pool v16](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)及 P1-4 其余上游 | 公开版本/来源条件未见能解除既有逐行身份与全训练集筛选问题的更新。 | 0 |

**结论：新增可用于严格官方 prompt 五折的外部行仍为 0。** 这不否定这些资料的探索价值；在取得独立原始调用与许可证据前，不把它们作为本项目的无泄漏增量，也不启动对应 GPU 训练。优先继续跟踪 kilupy 原始请求/响应记录和许可证一致性。
