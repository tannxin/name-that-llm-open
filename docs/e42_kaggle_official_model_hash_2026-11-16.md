# E42 Kaggle 官方 14B 底座文件核验

2026-11-16。此记录仅报告无评分的 CPU 文件完整性检查；未加载 GPU、未检查分类器效果，也未生成比赛提交。

证据来自 [私有 Kaggle Notebook 的保存输出](https://www.kaggle.com/code/tanxin357/probe-e42-t4-cpu-hash-only-v1/edit)（仅有访问权限的账号可查看）。检查对象是官方 Kaggle QwenLM Qwen 3 Transformers 14b-base V1，对照固定的 Qwen/Qwen3-14B-Base revision 0b0bd3732e2c374d483664439ea334928b65f304；锁定模型清单 SHA-256 为 f5ec051747d97321afc0b572b7fb2ea82806b29d9d03cbd906cc72b84e9e6e0b。

| 核验项 | 保存输出 |
| --- | ---: |
| 必需文件 | 15 |
| 大小及 SHA-256 均匹配 | 14 |
| 缺失或无法读取 | 0 |
| 已计算的 SHA-256 不匹配 | 0 |
| 大小不匹配 | tokenizer_config.json |
| 官方输入额外文件 | README.md |
| Adapter | 未挂载 |

由于 tokenizer_config.json 的大小不同，它不计入 14 个匹配文件，也不能据此认定官方输入是完整锁定的 15 文件底座。大小不符文件没有被报告为“SHA-256 不匹配”；两者是不同的核验结果。

接下来的无评分部署探针将只在逐文件验证跨输入的全部 15 个锁定哈希后，尝试使用官方输入中匹配的文件与单独提供的精确锁定配置。此检查尚未证明双 T4 显存、原始 logits 一致性、4,000 行吞吐、9 小时完整流程或提交可行性；当前不据此全量训练或提交。
