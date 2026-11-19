# E24 + E42 的 Kaggle 公榜成绩为 0.452

E24 加入 Qwen3-14B-Base 全量 Recovery1 后，Kaggle 公榜达到 **0.452**，比此前最佳 E24 + E39（8B）的 0.445 高 **0.007**，比 E24 的 0.434 高 **0.018**。

该版本使用在全部官方训练数据上训练的 E42 adapter，融合系数固定为 1。Notebook Version 1（`371364901`）于 2026 年 11 月 19 日 10:33:25 UTC 提交。Kaggle 提交详情随后显示 `Succeeded` 和公榜分数 0.452。[比赛提交记录](https://www.kaggle.com/competitions/name-that-llm/submissions)

| 模型组合 | 开发集五折平衡准确率 | Kaggle 公榜 |
|---|---:|---:|
| E24 | 0.425000 | 0.434 |
| E24 + E39（8B） | 0.443056 | 0.445 |
| E24 + E42（14B） | 0.443519 | 0.452 |

从 8B 版本到 14B 版本，五折平衡准确率增加 0.000463，公榜增加 0.007。两次评估的提升方向一致。[E39 五折与公榜记录](e39_full_official_recovery1_results.md) · [E42 五折审计](e42_qwen3_14b_base_recovery2_fivefold_results.md)
