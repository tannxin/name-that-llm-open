# E15：ModernBERT-base 监督微调与固定融合

## 设置

固定 `answerdotai/ModernBERT-base` revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`，使用服务器原始 `run_finetune_encoder.py` 的 18 类 mean-pooling 分类头。除编码器外，对照 E2d-R 保持相同的官方训练数据和五折 prompt 划分（120 个开发 prompt、2,160 条回答）、seed 42、256 tokens、batch 16、12 epochs、dropout 0.1、编码器学习率 `2e-5`、分类头 `1e-3`、AdamW、10% warmup、fp16 和固定末轮评估。训练集、划分及 E0 OOF 的 SHA-256 与 E2d-R 配置逐项核对一致。30 个锁定保留 prompt 未进入训练、分词或评分。

服务器为 Torch 2.6.0+cu118、Transformers 4.51.3，一张 RTX 3090（物理 GPU 1）。初次启动在首批前因 ModernBERT 默认调用 `torch.compile`、环境缺少可用 Triton 而失败，未产生 OOF。随后只设置 `TORCHDYNAMO_DISABLE=1` 使其使用 eager 实现，在独立目录 `outputs/e15_modernbert_base_12ep_eager/` 完成五折；预训练 checkpoint 的实际 revision 与请求 SHA 一致。五折训练约 802 秒，未保存折模型权重。失败启动的配置与日志单独留在 `outputs/e15_modernbert_base_12ep/`，没有混入成功运行结果。

## 开发集折外结果

| 折 | 原 RoBERTa 单模型 | ModernBERT 单模型 | E10：E0c＋RoBERTa | E0c＋ModernBERT | E0c＋RoBERTa＋ModernBERT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.319444 | 0.335648 | 0.368056 | 0.370370 | 0.405093 |
| 2 | 0.289352 | 0.328704 | 0.402778 | 0.402778 | 0.405093 |
| 3 | 0.361111 | 0.291667 | 0.449074 | 0.416667 | 0.442130 |
| 4 | 0.349537 | 0.358796 | 0.372685 | 0.377315 | 0.372685 |
| 5 | 0.349537 | 0.335648 | 0.405093 | 0.381944 | 0.412037 |
| **合并 OOF balanced accuracy** | **0.333796** | **0.330093** | **0.399537** | **0.389815** | **0.407407** |

融合均使用 E10 的固定规则：每路 18 类分数在各回答内减均值、除标准差，再等权相加，没有拟合温度或权重。所有比较检查 2,160 行的 `id,prompt_id,true_model,fold` 完全相同，并逐行重现已保存的 E10 预测。以 120 个 prompt 为单位做 20,000 次配对 bootstrap：

| 对照 | OOF 差值 | 95% 百分位区间 |
| --- | ---: | ---: |
| ModernBERT 单模型 − 原 RoBERTa | -0.003704 | [-0.030567, +0.023148] |
| E0c＋ModernBERT − E10 | -0.009722 | [-0.030093, +0.010648] |
| 三路等权融合 − E10 | +0.007870 | [-0.006481, +0.022222] |

ModernBERT 单模型家族正确率为 **0.679630**，家族正确条件下精确版本正确率为 **0.485695**；原 RoBERTa 分别为 **0.715741**、**0.466365**。E0c＋ModernBERT 为 **0.723611**、**0.538708**，E10 为 **0.756019**、**0.528475**。三路融合为 **0.751389**、**0.542206**。三路融合相对 E10 改变 518 个预测，其中 138 个由错变对、121 个由对变错，净增 17/2,160。

单独替换 RoBERTa 没有改善，三路等权融合显示一定互补性，但 +0.0079 的 prompt 配对区间跨零，且三路候选是在已有实验结果后追加分析，未调整多次选模偏差。**目前将三路融合保留为探索性候选，不据此重训全量包或提交 Kaggle。**

成功运行的配置、指标、折外分数与逐类召回保存在 Git 忽略的 `outputs/e15_modernbert_base_12ep_eager/`；固定两路对比在 `comparison_fixed_e10.json`，三路对比在 `three_way_fixed_e10.json`，三路逐行预测在 `three_way_fusion_oof.csv`。该目录及日志已从服务器复制到本地项目，OOF 文件 SHA-256 两端一致。锁定保留集与 Kaggle 隐藏标签均未使用。

在服务器项目根目录、相同环境与已缓存检查点下，可使用新的空输出目录复现训练：

```bash
TORCHDYNAMO_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 \
  python run_finetune_encoder.py \
  --model-id answerdotai/ModernBERT-base \
  --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --epochs 12 --pooling mean --max-length 256 --batch-size 16 \
  --no-save-checkpoints --output-dir outputs/e15_modernbert_base_12ep_reproduce
```
