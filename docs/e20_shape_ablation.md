# E20：固定 shape 字符特征消融

## 方案与复现边界

按[下一轮计划](next_round_plan_2026-11-06.md)预先固定的唯一候选，在原 E0c 的 word 1–2 gram、原文 char 3–5 gram、57 项 style 特征之外，只追加一块 shape char TF-IDF。shape 文本严格按公开 [V47 Notebook](https://www.kaggle.com/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble?scriptVersionId=365000313) 中可核对的表达式生成：

```python
re.sub(r'[^\W\d_]+', lambda m: 'X' if m[0].isupper() else 'x', text)
```

这里 `m[0]` 是**整个匹配到的字母串**，所以 `.isupper()` 判断整串是否大写；不是只判断首字母。连续 Unicode 字母替换成一个 `X` 或 `x`，数字、标点、下划线与空格保留。新块固定为 char 2–6 gram、`lowercase=False`、`sublinear_tf=True`、`min_df=2`、`max_features=60000`、`dtype=float32`、`FeatureUnion` 权重 0.5；原三块及 `LinearSVC(C=1, class_weight='balanced', dual=True, random_state=42)` 保持不变。未测试其他权重、n-gram 或形态替换规则。

服务器在项目目录使用原训练文件 SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`、原 E0c 脚本 SHA-256 `8bf5255471f263dae2f6ce94a84c59754702ed64bb4a080b05a0736dbc2d11ba`，以及 Python 3.10.12、scikit-learn 1.7.2、NumPy 2.2.6、pandas 2.3.3。先按既有 120 个开发 prompt、2,160 行、五个 prompt 分组折重新拟合原 E0c，**2,160 行预测与全部已保存的逐类 decision scores 精确相同，最大绝对差 0**；再在同折运行唯一 shape 模型。TF-IDF 词表和 style 标准化都只从当前折的训练 prompt 拟合。E20 未用其余 30 个 prompt 或 Kaggle 标签；这 30 个 prompt 在项目其他阶段已被用过，不能再称为未见保留集。

E16 对照沿用 [analyze_modernbert_seeds.py](../analyze_modernbert_seeds.py) 的配置和逐行核验：每路 18 类分数逐行 z 标准化，三个 ModernBERT seed 的标准化分数先平均为一路，再与 E0c、RoBERTa 各等权相加。E20 只将 E0c 的分数换为 shape E0c，其他 OOF 文件与权重不变。旧 E16 的 2,160 行融合预测逐行复现，原 0.409259 指标也复现。训练在服务器原环境完成；本地 scikit-learn 1.5.2 **只对保存的分数做对齐、argmax、指标和 bootstrap**，没有用它拟合 E20。

## 五折与总体结果

| 比较 | 原方案 BA | E20 BA | 差值 | 按 120 prompt 配对 bootstrap 的 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| E0c 单路 | 0.342593 | **0.353704** | **+0.011111** | [+0.001852, +0.020370] |
| E16 固定三路融合 | 0.409259 | **0.413426** | **+0.004167** | [-0.002778, +0.011111] |

Bootstrap 为 20,000 次、seed 42，以 prompt 为重采样单位。E16 融合有 37 行由错变对、28 行由对变错，净增 9/2,160 行。单路区间高于零，但项目已在同一开发集上开展多轮方法尝试，该区间未校正多重选择；融合区间仍覆盖零。

| 折 | E0c | shape E0c | 单路差值 | E16 | E16 换 shape | 融合差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.326389 | 0.324074 | -0.002315 | 0.391204 | 0.384259 | -0.006944 |
| 2 | 0.372685 | 0.368056 | -0.004630 | 0.425926 | 0.437500 | +0.011574 |
| 3 | 0.395833 | 0.405093 | +0.009259 | 0.435185 | 0.439815 | +0.004630 |
| 4 | 0.307870 | 0.324074 | +0.016204 | 0.375000 | 0.379630 | +0.004630 |
| 5 | 0.310185 | 0.347222 | +0.037037 | 0.418981 | 0.425926 | +0.006944 |

融合在 4/5 折提高，第 1 折下降。shape 单路在 3/5 折提高。所有折均为 24 个 prompt、432 行、每类 24 行，因此折内 BA 与准确率数值相等；总体每类各 120 行。

## 18 类召回

| 类别 | E0c | shape E0c | E16 | E16 换 shape | 融合差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.5` | .3667 | .3667 | .3333 | .3250 | -.0083 |
| `gpt-5.6-luna` | .2583 | .2667 | .4667 | .4750 | +.0083 |
| `gpt-5.6-sol` | .1417 | .1583 | .1917 | .1917 | .0000 |
| `gpt-5.6-terra` | .1667 | .1917 | .3000 | .2833 | -.0167 |
| `gpt-6-astra` | .6833 | .6917 | .7417 | .7333 | -.0083 |
| `gemini-3.1-pro-preview` | .7583 | .7667 | .8000 | .8000 | .0000 |
| `gemini-3.6-flash` | .3167 | .3250 | .5667 | .5750 | +.0083 |
| `gemini-3.7-flash` | .3417 | .3333 | .3250 | .3250 | .0000 |
| `gemini-3.8-flash` | .3000 | .3000 | .3000 | .3083 | +.0083 |
| `claude-fable-5` | .0333 | .0833 | .1750 | .2083 | +.0333 |
| `claude-opus-4-6` | .5833 | .5667 | .5833 | .6083 | +.0250 |
| `claude-opus-4-8` | .1833 | .2250 | .3750 | .4083 | +.0333 |
| `claude-opus-5` | .5833 | .5750 | .5833 | .5833 | .0000 |
| `claude-sonnet-5` | .3667 | .4250 | .3500 | .3500 | .0000 |
| `qwen3.8-max` | .5000 | .5000 | .5000 | .5083 | +.0083 |
| `glm-5.3` | .1333 | .1667 | .2083 | .2167 | +.0083 |
| `glm-5.3-flash` | .1167 | .1333 | .2917 | .2500 | **-.0417** |
| `kimi-k3` | .3333 | .2917 | .2750 | .2917 | +.0167 |

融合最大单类损失是 `glm-5.3-flash` 的 5/120；最大收益为 `claude-fable-5` 和 `claude-opus-4-8`，各 +4/120。单路的 `kimi-k3` 下降 5/120，`claude-sonnet-5` 增加 7/120。逐行预测、精确召回值和原始分数见下方本地产物。

从逐行变化看，shape 使 E0c 单路 303 个预测改变，其中 71 个由错变对、47 个由对变错；这 71 个单路新增正确样本中，原 E16 融合已答对 32 个，单独三种子 ModernBERT 已答对 25 个。原 E16 尚未答对的 39 个样本中，换入 shape 后只有 3 个转为正确。融合自身有 125 个预测改变，其中 37 个由错变对、28 个由对变错，其余为错类之间转移。这些观察说明 shape 主要提高 E0c 单路，对现有融合的新增贡献有限；其中一部分单路收益已被原融合覆盖。原融合同时含 RoBERTa 和 ModernBERT，不能从这些计数单独认定重叠来自 ModernBERT。逐类上，`claude-fable-5` 和 `claude-opus-4-8` 的小幅融合收益伴随 `glm-5.3-flash` 的损失，平均后的净增量因此较小。

## 决策与复现

预先写下的晋级要求是固定融合相对 E16 **至少 +0.01**、至少 4/5 折不降、没有单类召回下降 ≥0.05。E20 满足后两项，融合增益只有 **+0.004167**，且 prompt 配对区间跨零；**本轮不替换 E16 主方案，也不追加 Kaggle 提交**。这个结果说明 shape 特征在传统 E0c 单路有互补信息，但进入既有神经集成后新增信息较小。公开 V47 的成绩还混合外部数据和多种模型，不能据此把其总分归因给 shape 单块。

独立脚本：[run_e20_shape_ablation.py](../run_e20_shape_ablation.py)。在服务器原环境执行：

```bash
cd /path/to/namethatllm
python -u run_e20_shape_ablation.py fit --output-dir outputs/e20_shape_fit
```

将该输出目录的 `decision_scores.csv`、`oof_predictions.csv`、`fit_metrics.json` 复制回本地项目后执行：

```bash
python run_e20_shape_ablation.py analyze --output-dir outputs/e20_shape_analysis
```

本地 Git 忽略产物 `outputs/e20_shape_fit/` 与 `outputs/e20_shape_analysis/` 保留完整配置、逐行分数与对照。`outputs/e20_shape_fit/decision_scores.csv` 在服务器和本地的 SHA-256 均为 `a2a80c13519aadd417712fea4bb3801fefc33a814b9cdf02829f10067416fd97`；原 E0c 和 RoBERTa 输入分数哈希分别为 `baa3466481967ecd82483564286c697f6011b1c73bab7635a17393e0aeefb27e` 和 `86c6621058dc630b690bc21a58b5767872466c416395c02c327491b8b16c80cd`。三个 ModernBERT OOF 哈希与 [E16 原记录](e16_modernbert_seeds.md)一致。
