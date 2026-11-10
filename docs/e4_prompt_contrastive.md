# E4：同题难负例对比学习

## 设计

在 E0 固定的 120 个开发 prompt、2,160 条回答上，沿用五折 prompt 分组划分。每折训练 96 个 prompt，验证 24 个 prompt；30 个最终保留 prompt 不进入分词、训练或评分。三个条件使用相同的预训练权重、初始化种子、每轮 prompt 配对顺序、分类头和训练日程，只改变损失函数与对比损失的负例集合：

| 条件 | 损失 |
| --- | --- |
| `ce_only` | 18 类加权交叉熵 |
| `ce_supcon_crossprompt` | 相同交叉熵 + `0.2 × SupCon`，只使用跨 prompt 负例 |
| `ce_supcon` | 相同交叉熵 + `0.2 × SupCon`，再加入同题难负例 |

每个训练 batch 从两个完整 prompt 各取 18 个模型回答，共 36 条。对每条回答，同模型在另一 prompt 的回答是唯一正例。跨 prompt 对照有 17 个其他模型负例；难负例条件额外纳入同题的 17 个其他模型回答，共 34 个负例。SupCon 温度为 0.1。编码器为固定版本 `sentence-transformers/all-MiniLM-L6-v2`（`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`）；attention-mask 平均池化的表示输入 dropout 0.1 + 线性分类头，对比损失使用该表示的 L2 归一化副本。

三组均固定训练 12 轮、最大长度 256 tokens、batch 36、seed 42；编码器学习率 `2e-5`、分类头 `1e-3`，AdamW、weight decay 0.01、10% warmup 后线性衰减、fp16 自动混合精度。GradScaler 跳过优化器更新时不推进学习率调度器。只用第 12 轮模型评估，不依照验证集选择轮数。运行使用一张 RTX 3090。训练集、划分及 E0 OOF 的 SHA-256 分别为 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`、`653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15`、`698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa`。三组的 `batch_pairs.csv` SHA-256 完全相同，配置和库版本分别保存在 `outputs/e4_prompt_contrastive/` 与 `outputs/e4_prompt_contrastive_crossprompt/`。

## 开发集结果

| 折 | CE | CE + 跨题负例 | CE + 跨题及同题负例 |
| --- | ---: | ---: | ---: |
| 1 | 0.277778 | 0.275463 | 0.277778 |
| 2 | 0.289352 | 0.229167 | 0.268519 |
| 3 | 0.268519 | 0.273148 | 0.252315 |
| 4 | 0.296296 | 0.289352 | 0.303241 |
| 5 | 0.273148 | 0.268519 | 0.273148 |
| **合并 OOF balanced accuracy** | **0.281019** | **0.267130** | **0.275000** |

三组分别正确分类 607、577、594 / 2,160 条。按六个模型家族统计，家族识别准确率分别为 0.6412、0.6282、0.6394；在家族识别正确的回答中，精确版本识别率分别为 0.4383、0.4252、0.4301。加入任一种对比损失都没有改善本轮的家族内版本区分。

逐行 OOF 的 `id,prompt_id,true_model,fold` 完全对齐，且每个开发 prompt 都包含全部 18 类。以 120 个 prompt 为重采样单位、随机种子 42 做 20,000 次配对 bootstrap，得到以下 balanced accuracy 差值与 95% 百分位区间：

| 对比 | 差值 | 区间 |
| --- | ---: | ---: |
| 跨题负例 SupCon − CE | -0.013889 | [-0.027315, -0.000926] |
| 加入同题难负例 − 只用跨题负例 | +0.007870 | [-0.009722, 0.025463] |
| 同题难负例 SupCon − CE | -0.006019 | [-0.021759, 0.009722] |

实验计划原本包括普通对比学习对照；具体的“仅跨题负例”掩码是在前两组结果出现后补做，沿用了相同的损失权重、温度及训练日程。这些区间只作探索性描述，不能当作多组比较后校正的确认性显著性结论。同题难负例相对只用跨题负例的分数稍高，但区间跨零；两种对比损失都低于 CE，因此当前配置不进入最终候选。这不排除其他编码器或损失权重下对比学习可能有效。

三组共十五次折训练每折有 1–2 次 fp16 梯度溢出引起的优化器跳步，脚本已记录跳步并同步跳过调度器更新。CE 与同题难负例组的指标和逐样本预测见 `outputs/e4_prompt_contrastive/{ce_only,ce_supcon}/`，只用跨题负例组见 `outputs/e4_prompt_contrastive_crossprompt/ce_supcon_crossprompt/`。这些输出及模型权重不提交至 Git；服务器保留了各折权重。

复现命令：

```bash
python run_prompt_contrastive.py --train data/train.csv --split-manifest outputs/e0_tfidf_svm/split_manifest.csv --e0-oof outputs/e0_tfidf_svm/oof_predictions.csv --output-dir outputs/e4_prompt_contrastive_reproduce
python compare_oof.py --reference outputs/e4_prompt_contrastive_reproduce/ce_only/oof_predictions.csv --candidate outputs/e4_prompt_contrastive_reproduce/ce_supcon_crossprompt/oof_predictions.csv
python compare_oof.py --reference outputs/e4_prompt_contrastive_reproduce/ce_supcon_crossprompt/oof_predictions.csv --candidate outputs/e4_prompt_contrastive_reproduce/ce_supcon/oof_predictions.csv
```
