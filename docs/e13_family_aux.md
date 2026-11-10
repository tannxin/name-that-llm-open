# E13：18 类输出上的软家族辅助损失

## 问题与预设

比赛要求预测 18 个精确模型，评分为各模型召回率的平均值。18 类按官方提供方可分为六个家族：

| 家族 | 精确标签 | 类数 |
| --- | --- | ---: |
| GPT | `gpt-5.5`、`gpt-5.6-luna`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-6-astra` | 5 |
| Gemini | `gemini-3.1-pro-preview`、`gemini-3.6-flash`、`gemini-3.7-flash`、`gemini-3.8-flash` | 4 |
| Claude | `claude-fable-5`、`claude-opus-4-6`、`claude-opus-4-8`、`claude-opus-5`、`claude-sonnet-5` | 5 |
| Qwen | `qwen3.8-max` | 1 |
| GLM | `glm-5.3`、`glm-5.3-flash` | 2 |
| Kimi | `kimi-k3` | 1 |

这次保持原 RoBERTa 的 18 类分类头与推理规则，只在训练时增加家族损失。若 18 个 logits 为 $z$，真标签所属家族为 $f(y)$，则家族概率为 $q_f=\sum_{j\in f}\operatorname{softmax}(z)_j$，总损失为

$$L=L_{18\text{类加权交叉熵}}+0.2\,[-\log q_{f(y)}].$$

`0.2` 在训练前固定，没有在开发集上搜索权重。代码用各家族 logits 的 `logsumexp` 实现该概率；不先选家族，也不训练单独的 6 类输出头。GPU 1 上的小张量检查确认了家族概率等于对应 18 类概率之和，且反向梯度有限。

## 与 E2d-R 相同的设置

官方训练数据 2,699 行、150 个 prompt；仍只用原开发集 120 个 prompt、2,160 行及原五折 prompt 分组。30 个锁定保留 prompt 从未进入本实验的分词、训练、调参或评分。编码器固定为 `FacebookAI/roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`；mean pooling、dropout 0.1、最大 256 tokens、batch 16、12 epochs、seed 42、编码器学习率 `2e-5`、分类头 `1e-3`、AdamW、weight decay 0.01、10% warmup 后线性衰减、fp16。GradScaler 跳过优化器更新时不推进学习率调度器，只在第 12 轮评估。与 E2d-R 的 `run_config.json` 中模型、数据哈希、划分哈希、训练日程和随机种子逐项比对一致，唯一训练目标差别为家族损失。新实验没有保存五折权重，只保存折外预测、配置和指标；原 E2d-R 及当前提交包未改动。

训练脚本为 [`run_finetune_encoder.py`](../run_finetune_encoder.py) 的 `--family-loss-weight 0.2`；服务器上以独立文件名 `run_finetune_encoder_family_aux.py` 运行，原服务器脚本未覆盖。完整输出在 Git 忽略的 `outputs/e13_roberta_family_aux_02/`；只读比较脚本为 [`analyze_family_aux.py`](../analyze_family_aux.py)，结果在 `outputs/e13_roberta_family_aux_02/fusion_analysis/`。比较前严格检查所有折外行的 `id,prompt_id,true_model,fold`、原训练配置及保存的 E10 预测能被原分数融合规则逐行重现。

## 五折开发集结果

| 折 | 原 RoBERTa 18 类 | 家族损失 RoBERTa 18 类 | 原 E10 融合 | 家族损失模型＋E0c 固定融合 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.319444 | 0.326389 | 0.368056 | 0.365741 |
| 2 | 0.289352 | 0.303241 | 0.402778 | 0.402778 |
| 3 | 0.361111 | 0.340278 | 0.449074 | 0.409722 |
| 4 | 0.349537 | 0.349537 | 0.372685 | 0.388889 |
| 5 | 0.349537 | 0.349537 | 0.405093 | 0.418981 |
| **合并 OOF** | **0.333796** | **0.333796** | **0.399537** | **0.397222** |

两组融合均使用 E10 已固定的规则：E0c 和 RoBERTa 各自的 18 类分数在每条回答内减均值、除标准差，再等权相加；没有按这轮标签调融合权重。新 RoBERTa 与原 RoBERTa 各答对 721/2,160；新融合答对 858/2,160，比 E10 的 863/2,160 少 5 条。以 120 个 prompt 为单位做 20,000 次配对 bootstrap，新减旧的 95% 百分位区间为：直接 RoBERTa **[-0.012500, +0.012037]**；固定融合 **[-0.013889, +0.009259]**。这些区间未校正此前多轮尝试的选模偏差。

| 方法 | 家族正确率 | 家族正确时的精确标签正确率 |
| --- | ---: | ---: |
| E2d-R 直接 RoBERTa | 0.715741 | 0.466365 |
| 新 RoBERTa 家族损失 | 0.724074 | 0.460997 |
| E10 原固定融合 | 0.756019 | 0.528475 |
| 新固定融合 | 0.753704 | 0.527027 |

家族损失让 RoBERTa 本身多判对 18 条的家族，但家族内版本正确率下降，最终 18 类分数持平；与 E0c 融合后两个层级都略低于 E10。

## 与硬式门控及后续取舍

先前 E7a 用相同 E0c 特征先判家族、再只在该家族判版本，18 类 OOF 为 **0.3250**，低于平坦 E0c 的 **0.3426**，且五折均低于 E0c。E7a 家族判断正确率为 **0.7120**，但家族判错时下一级无法纠正。进一步只读诊断把 E7a 的折外家族预测套到 E10 的折外融合分数、仅在预测家族内取最高标签，得到 **0.3519**，低于不加门控的 E10 **0.3995**；这不是一次重新训练，也不是提交候选。

本次固定的软家族损失没有提供可靠提升，故不据此重训全量包或提交 Kaggle，E10 继续作为当前参照。该结果只检验了这一种损失、权重及 RoBERTa 日程；它不能排除共享编码器双头、校准概率软门控或其他编码器上的层级方法。若以后尝试，应继续以 18 类 balanced accuracy 和逐类召回为主要判断依据，并保持按 prompt 分组、避免复用已锁定的保留集。

发布时间可以作为描述模型关系的外部信息，但不适合作为硬路由层。测试输入只有 `id,text`，没有可用的生成时间；而 `gpt-5.6-luna/sol/terra` 是同一版本号下的不同档位，不能仅凭标签名排成时间树。Qwen 和 Kimi 各只有一个目标标签，也没有更深的版本分支。若要增加第三层，应先核实版本谱系，再检验这一层能否改善 18 类 OOF；优先参考同题输出距离、定向混淆及跨 prompt 的稳定性，见[类别配对诊断](pair_geometry_diagnostic.md)。

复现命令（在有 CUDA 和已缓存模型权重的服务器项目根目录运行，输出目录需为空）：

```bash
CUDA_VISIBLE_DEVICES=1 python run_finetune_encoder.py --model-id FacebookAI/roberta-base --revision e2da8e2f811d1448a5b465c236feacd80ffbac7b --epochs 12 --pooling mean --max-length 256 --batch-size 16 --seed 42 --family-loss-weight 0.2 --no-save-checkpoints --output-dir outputs/e13_roberta_family_aux_02_reproduce
python analyze_family_aux.py --family-aux-scores outputs/e13_roberta_family_aux_02_reproduce/oof_predictions.csv --output-dir outputs/e13_roberta_family_aux_02_reproduce/fusion_analysis
```

比赛标签与评分方式见 [Kaggle 官方概述](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)。
