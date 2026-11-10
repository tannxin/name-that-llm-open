# E16：ModernBERT 随机种子复核与预定三种子平均

## 固定方案与核验

按[第三轮计划](round3_plan.md)第 1 项，将 E15 的 `answerdotai/ModernBERT-base` 固定在 revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`，新跑 seed 17 和 73，原 E15 seed 42 保留。三次运行都使用同一批 120 个开发 prompt、2,160 行、五折 prompt 分组、256 tokens、batch 16、12 个固定 epoch、mean pooling、相同学习率和优化器；只更改训练随机种子。原始训练数据、划分清单、E0 OOF 的 SHA-256 与 E15 完全一致。每次都在末轮评估，未按验证折挑选 epoch。30 个锁定保留 prompt、Kaggle 测试集均未用于本轮选择。

在服务器物理 GPU 1、2 上分别运行 seed 17、73，使用与 E15 相同的 `TORCHDYNAMO_DISABLE=1` eager 路径和原始 `run_finetune_encoder.py`。两个独立输出目录分别是 `outputs/e16_modernbert_seed17_eager/`、`outputs/e16_modernbert_seed73_eager/`；预训练权重实际 revision 均与指定 SHA 相同，未保存模型权重。完整训练配置、五折指标和折外分数已复制到本地同名 Git 忽略目录，OOF SHA-256 与服务器逐项一致：seed 17 为 `da391ea077c200e9735237183cf02f7d57b5b809685b437d37b5e0ef3346c189`，seed 73 为 `23de22abb848effb851fd2f079283d6c726a2d8f9c339920a227d4b981989ba8`。

分析脚本 [analyze_modernbert_seeds.py](../analyze_modernbert_seeds.py)逐项核对三次运行除 seed 外的配置、2,160 行的 `id,prompt_id,true_model,fold`、各模型分数与保存预测的一致性；它逐行重现 E10 和原 E15 三路预测。所有融合沿用 E10 的无参数规则：每路 18 类分数在每条回答内减均值、除标准差后等权相加。**预先固定的三 seed 集成**先分别标准化三个 ModernBERT 分数矩阵，取均值作为一条 ModernBERT 路，再与 E0c、RoBERTa 两路各等权相加；不能先平均原始 logits，也不能把三个 seed 当作独立的三条融合路。

## 开发集 OOF

| ModernBERT 来源 | 单模型 | E0c＋ModernBERT | E0c＋RoBERTa＋ModernBERT | 三路相对 E10 | Prompt 配对 bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed 42，原 E15 | 0.330093 | 0.389815 | 0.407407 | +0.007870 | [-0.006481, +0.022222] |
| seed 17 | 0.337037 | 0.391667 | 0.408796 | +0.009259 | [-0.006019, +0.024086] |
| seed 73 | 0.322685 | 0.388426 | 0.407407 | +0.007870 | [-0.007407, +0.022685] |
| **三 seed 预定平均** | **0.359722** | **0.399537** | **0.409259** | **+0.009722** | **[-0.004630, +0.024074]** |

E10 的 18 类 balanced accuracy 为 **0.399537**。三个单 seed 的三路融合都比 E10 高约 0.008–0.009，预定集成高 0.0097；但四个 prompt 配对区间都跨零。bootstrap 每次重采样 120 个 prompt，20,000 次、seed 42；三个训练 seed 共用同样的训练行与验证 prompt，不能当成三个独立验证集。

| 折 | E10 | seed 42 三路 | seed 17 三路 | seed 73 三路 | 三 seed 平均三路 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.368056 | 0.405093 | 0.384259 | 0.388889 | 0.391204 |
| 2 | 0.402778 | 0.405093 | 0.421296 | 0.418981 | 0.425926 |
| 3 | 0.449074 | 0.442130 | 0.423611 | 0.437500 | 0.435185 |
| 4 | 0.372685 | 0.372685 | 0.407407 | 0.379630 | 0.375000 |
| 5 | 0.405093 | 0.412037 | 0.407407 | 0.412037 | 0.418981 |

seed 42 是 3 折提高、1 折相等、1 折下降；seed 17、73 和三 seed 平均都是 **4 折提高、1 折下降**。三 seed 平均相对 E10 在 2,160 行中有 125 个由错变对、104 个由对变错，净增 21 行。家族正确率从 E10 的 **1633/2160 = 0.7560** 略降为 **1624/2160 = 0.7519**，但家族正确条件下版本正确率从 **863/1633 = 0.5285** 升为 **884/1624 = 0.5443**；新增的精确分类收益主要来自家族内部版本判别。

## 逐类召回与同题稳定性

下面每类各有 120 个开发样本；完整的三 seed 逐类召回见 Git 忽略的 `outputs/e16_modernbert_seed_analysis_checked/class_metrics.csv`。

| 类别 | E10 | 三 seed 平均三路 | 差值 |
| --- | ---: | ---: | ---: |
| `gpt-5.5` | 0.3000 | 0.3333 | +0.0333 |
| `gpt-5.6-luna` | 0.5250 | 0.4667 | **-0.0583** |
| `gpt-5.6-sol` | 0.1667 | 0.1917 | +0.0250 |
| `gpt-5.6-terra` | 0.2583 | 0.3000 | +0.0417 |
| `gpt-6-astra` | 0.7333 | 0.7417 | +0.0083 |
| `gemini-3.1-pro-preview` | 0.8167 | 0.8000 | -0.0167 |
| `gemini-3.6-flash` | 0.5583 | 0.5667 | +0.0083 |
| `gemini-3.7-flash` | 0.3083 | 0.3250 | +0.0167 |
| `gemini-3.8-flash` | 0.3000 | 0.3000 | 0.0000 |
| `claude-fable-5` | 0.1833 | 0.1750 | -0.0083 |
| `claude-opus-4-6` | 0.6250 | 0.5833 | **-0.0417** |
| `claude-opus-4-8` | 0.3667 | 0.3750 | +0.0083 |
| `claude-opus-5` | 0.5583 | 0.5833 | +0.0250 |
| `claude-sonnet-5` | 0.3167 | 0.3500 | +0.0333 |
| `qwen3.8-max` | 0.5000 | 0.5000 | 0.0000 |
| `glm-5.3` | 0.1917 | 0.2083 | +0.0167 |
| `glm-5.3-flash` | 0.2500 | 0.2917 | +0.0417 |
| `kimi-k3` | 0.2333 | 0.2750 | +0.0417 |

18 类中 **12 类提高、4 类下降、2 类不变**。最大下降为 `gpt-5.6-luna` 的 7/120，次之为 `claude-opus-4-6` 的 5/120；未出现单类召回崩塌，但这两个类别的损失应在外部验证前记为明确风险。

每个 prompt 有 18 个样本。相对 E10，三 seed 平均使 **49 个 prompt 提高、36 个不变、35 个下降**。其中 21 个 prompt 在三个单 seed 三路预测中均下降，且三 seed 的逐 prompt 净增量两两相关约 **0.54–0.57**；因此模型平均不能消除这批共有错误。最差的 `p015` 在 E10 为 9/18 正确，三个单 seed 三路均为 5/18，集成为 4/18。完整逐 prompt 计数见 `outputs/e16_modernbert_seed_analysis_checked/prompt_metrics.csv`。这些是开发集诊断，未用来选择融合权重或屏蔽 prompt。

## 晋级判断与复现

按[事先写下的门槛](round3_plan.md)，三个 seed 的三路融合方向一致，预定集成在多数折提高，也没有严重的单类召回坍塌，**可以进入一次预先锁定的 Kaggle 外部验证**。这不等于已证明泛化增益：集成的 prompt 配对区间仍覆盖零，部分 prompt 在全部 seed 下系统性变差，且项目已经试过多个方法，区间未校正多次选模的乐观偏差。下一步需先在全部 150 个训练 prompt 上分别重新拟合三个 ModernBERT、E0c 和 RoBERTa，完成离线 Notebook 的约 4,000 行模拟推理、内存与九小时限制检查，然后只提交一次预定三 seed 规则。保留集不再用于调参。

服务器项目根目录下，两个新种子的训练命令分别把 `--seed` 设为 `17`、`73`，并使用各自全新的空输出目录；其余参数与 E15 相同：

```bash
TORCHDYNAMO_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 \
  python run_finetune_encoder.py \
  --model-id answerdotai/ModernBERT-base \
  --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --epochs 12 --pooling mean --max-length 256 --batch-size 16 \
  --seed 17 --no-save-checkpoints \
  --output-dir outputs/e16_modernbert_seed17_reproduce
```

对 seed 73，仅把命令中的 `--seed 17` 与输出目录分别换成 `--seed 73` 和新的 `outputs/e16_modernbert_seed73_reproduce`。本地分析使用：

```bash
python analyze_modernbert_seeds.py --output-dir outputs/e16_modernbert_seed_analysis_reproduce
```
