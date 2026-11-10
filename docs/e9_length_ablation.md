# E9：RoBERTa 输入长度消融

2026-11-05。目标是检验原 E2d-R 的 256-token 截断是否限制了未见 prompt 上的 18 类识别。只使用固定的 120 个开发 prompt、2,160 条回答和原有五折；30 个 prompt 的保留集未被分词、训练或评分。

## 截断审计

用原训练的 `FacebookAI/roberta-base` 固定 revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` tokenizer，计入特殊 token，先不截断再统计长度。`tokenizer.json` SHA-256 为 `847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e`。训练集、split manifest、E0 OOF 哈希分别与 E2d-R 的运行配置一致：`40a22ce8...b21de77707`、`653c23b1...a9955ffd15`、`698265e9...c3528fa`；逐行 ID、标签和五折分配也与 RoBERTa OOF 一致。

| 统计 | token 数或比例 |
| --- | ---: |
| 长度中位数 / P90 / P95 / P99 / 最大 | 132.5 / 264.1 / 288.0 / 318.4 / 387 |
| 超过 256 tokens | 256 / 2,160（11.85%） |
| 超过 512 tokens | 0 / 2,160 |
| 被 256 截断的回答：超出部分中位数 / P95 / 最大 | 26 / 70 / 131 tokens |

每类均有 120 条开发回答。256-token 截断最多的类别是 `claude-opus-4-8`（22）、`claude-sonnet-5`（20）与 `gpt-5.6-sol`（20），最少是 `gpt-5.5`（9）。旧 RoBERTa OOF 在截断组的正确率为 39.84%，未截断组为 32.51%；组别的类别及 prompt 构成不同，这个观察不能解释为截断提高准确率。完整逐类计数在 `outputs/truncation_audit/by_class.csv`，审计输入哈希和分位数在同目录的 `audit.json`。

## 固定对照

只把 `max_length` 从 256 改为 512。原 `run_finetune_encoder.py` 脚本哈希相同，仍用 12 轮、batch 16、seed 42、相同五折、RoBERTa revision、类别权重、学习率、dropout、warmup 和原有的“仅成功优化器更新后推进调度器”逻辑。512 能完整保留全部开发回答。新结果写入独立目录，未覆盖 E2d-R。GPU 2 上的五折各约 122 秒。

| 折 | RoBERTa 256 | RoBERTa 512 | E0c＋256 rank | E0c＋512 rank | E0c＋256 z-score | E0c＋512 z-score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3194 | 0.3356 | 0.3727 | 0.3681 | 0.3681 | 0.3704 |
| 2 | 0.2894 | 0.3264 | 0.3819 | 0.3843 | 0.4028 | 0.3935 |
| 3 | 0.3611 | 0.3449 | 0.4329 | 0.4190 | 0.4491 | 0.4282 |
| 4 | 0.3495 | 0.3333 | 0.3542 | 0.3727 | 0.3727 | 0.3866 |
| 5 | 0.3495 | 0.3333 | 0.3750 | 0.3750 | 0.4051 | 0.4074 |
| **Pooled** | **0.333796** | **0.334722** | **0.383333** | **0.383796** | **0.399537** | **0.397222** |

512 减 256 的 pooled balanced accuracy 变化及按 `prompt_id` 配对的 20,000 次 bootstrap 95% 分位数区间：

| 方法 | 差值 | 95% 区间 |
| --- | ---: | ---: |
| RoBERTa 单模型 | +0.000926 | [−0.015741, +0.017593] |
| E0c＋RoBERTa 等权 rank | +0.000463 | [−0.011111, +0.012037] |
| E0c＋RoBERTa 逐行 z-score 等权 | −0.002315 | [−0.014352, +0.009259] |

所有区间都跨 0。z-score 规则对每条回答的两路 18 类分数分别减去该行均值、除以该行标准差，再等权相加；没有在 OOF 标签上拟合权重、温度或校准器。相比 rank，512 的 z-score 融合为 0.397222，但仍低于 256 的 0.399537。

下表是 512 相对 256 的逐类召回率变化，单位为百分点。正负变化在不同类别间抵消，未显示稳定改善。

| 类别 | 单模型 | rank 融合 | z-score 融合 |
| --- | ---: | ---: | ---: |
| gpt-5.5 | −5.0 | −1.7 | −5.0 |
| gpt-5.6-luna | −0.8 | −1.7 | 0.0 |
| gpt-5.6-sol | −3.3 | +2.5 | +0.8 |
| gpt-5.6-terra | −1.7 | +4.2 | +0.8 |
| gpt-6-astra | −0.8 | −0.8 | −2.5 |
| gemini-3.1-pro-preview | −4.2 | −2.5 | −4.2 |
| gemini-3.6-flash | +5.0 | −0.8 | +2.5 |
| gemini-3.7-flash | −0.8 | −4.2 | 0.0 |
| gemini-3.8-flash | −1.7 | −0.8 | −2.5 |
| claude-fable-5 | +7.5 | −0.8 | −4.2 |
| claude-opus-4-6 | −0.8 | +2.5 | +5.0 |
| claude-opus-4-8 | +3.3 | 0.0 | +2.5 |
| claude-opus-5 | −2.5 | +1.7 | −4.2 |
| claude-sonnet-5 | −2.5 | +4.2 | +8.3 |
| qwen3.8-max | +5.0 | 0.0 | −1.7 |
| glm-5.3 | +5.0 | 0.0 | +0.8 |
| glm-5.3-flash | −2.5 | 0.0 | +1.7 |
| kimi-k3 | +2.5 | −0.8 | −2.5 |

**结论：512 不晋级。** 它完整消除了截断，但单模型及 rank 融合增益不到 0.001，z-score 融合略降；不据此重训全量模型或提交 Kaggle。首尾拼接 256 仍是另一种截取策略，当前证据不足以将它排在更有希望的特征和融合实验之前。

## 复现

在项目根目录，使用与 E2d-R 相同的 Python 环境，并令 `MODEL_CACHE` 指向含上述固定 revision 的 Hugging Face cache。原缓存 tokenizer 文件哈希须与上文一致。以下是完整训练及融合参数；`CUDA_VISIBLE_DEVICES=2` 使脚本内的 `cuda:0` 对应物理 GPU 2。`outputs/` 不进入 Git。

```bash
export HF_HOME="$MODEL_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python audit_roberta_length.py \
  --root . \
  --tokenizer-dir "$HF_HOME/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b" \
  --output-dir outputs/truncation_audit

CUDA_VISIBLE_DEVICES=2 python run_finetune_encoder.py \
  --train data/train.csv \
  --model-id FacebookAI/roberta-base \
  --revision e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --split-manifest outputs/e0_tfidf_svm/split_manifest.csv \
  --e0-oof outputs/e0_tfidf_svm/oof_predictions.csv \
  --output-dir outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix_512 \
  --epochs 12 --learning-rate 2e-5 --head-learning-rate 1e-3 \
  --max-length 512 --batch-size 16 --seed 42 \
  --warmup-ratio 0.1 --weight-decay 0.01 --dropout 0.1 --pooling mean

python run_fixed_blend.py \
  --left outputs/e0c_tfidf_plus_style/decision_scores.csv \
  --right outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix_512/oof_predictions.csv \
  --output-dir outputs/e7_blend_e0c_roberta_ft_scalerfix_512

python run_score_normalization.py \
  --left outputs/e0c_tfidf_plus_style/decision_scores.csv \
  --right outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix_512/oof_predictions.csv \
  --reference outputs/e7_blend_e0c_roberta_ft_scalerfix_512/oof_predictions.csv \
  --output-dir outputs/e10_fusion_normalization_roberta_512

python run_score_normalization.py \
  --left outputs/e0c_tfidf_plus_style/decision_scores.csv \
  --right outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv \
  --reference outputs/e7_blend_e0c_roberta_ft_scalerfix/oof_predictions.csv \
  --output-dir outputs/e10_fusion_normalization

python compare_oof.py \
  --reference outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix/oof_predictions.csv \
  --candidate outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix_512/oof_predictions.csv \
  --output outputs/e2_finetune/roberta_base_meanpool_12ep_scalerfix_512/comparison_vs_256.json

python compare_oof.py \
  --reference outputs/e7_blend_e0c_roberta_ft_scalerfix/oof_predictions.csv \
  --candidate outputs/e7_blend_e0c_roberta_ft_scalerfix_512/oof_predictions.csv \
  --output outputs/e7_blend_e0c_roberta_ft_scalerfix_512/comparison_vs_256.json

python - <<'PY'
from pathlib import Path
import pandas as pd
for length, directory in (
    (256, "outputs/e10_fusion_normalization"),
    (512, "outputs/e10_fusion_normalization_roberta_512"),
):
    frame = pd.read_csv(Path(directory) / "oof_predictions.csv", dtype=str)
    frame["pred_model"] = frame["zscore_pred_model"]
    frame[["id", "prompt_id", "true_model", "fold", "pred_model"]].to_csv(
        f"outputs/truncation_audit/zscore{length}_oof.csv", index=False
    )
PY

python compare_oof.py \
  --reference outputs/truncation_audit/zscore256_oof.csv \
  --candidate outputs/truncation_audit/zscore512_oof.csv \
  --output outputs/truncation_audit/zscore_comparison_vs_256.json
```

其中两个 z-score 输出均由同一脚本以显式输入和 reference 生成。配对比较把各自的 `zscore_pred_model` 作为 `pred_model` 后交给 `compare_oof.py`；原始逐行预测和完整比较结果分别保存在对应输出目录及 `outputs/truncation_audit/zscore_comparison_vs_256.json`。
