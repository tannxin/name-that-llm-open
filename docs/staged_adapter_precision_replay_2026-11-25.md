# Five-fold loading precision comparison for E24 + Qwen3-14B

The original loading method and the method that preserves the saved FP32 classification head each classified **1,042 of 2,160 responses correctly**. Both achieved **0.482407 balanced accuracy**.

Both methods used the same five Qwen adapters trained first on external data and then on official data. The E24 fusion weights, automatic question groups and within-group soft matching were fixed. Each fold evaluated 24 held-out development questions, giving 120 questions across five folds.

The two methods agreed on **2,159 of 2,160** predicted labels. On the remaining response they chose different incorrect labels. The number correct was identical in every fold, and recall was identical for all 18 classes. The balanced accuracy difference was **0.000000**, with a 95% interval of **[0.000000, 0.000000]** from 20,000 paired bootstrap resamples of questions.

The original pipeline was selected on these same development folds. The comparison provides an offline result for the revised loading method. Its hidden-test score remains unmeasured.

## Kaggle deployment checks

On two T4 GPUs with internet access disabled, the full Qwen3-14B adapter preserved its saved FP32 classification head exactly. Across three fixed training texts and 18 classes, the largest absolute logit difference was **0.004915** against the pre-save reference and **0.000552** against the revised 3090 reload. All three top predictions matched both references, within the fixed absolute tolerance of 0.05 (relative tolerance zero). Kaggle reported **705 seconds** for the run, including 115.4 seconds for model loading and the check. The original reload had exceeded the same tolerance on the 3090.

The official three-row sample also completed on two T4 GPUs in **1,326.5 seconds**. An independent read of the output Parquet confirmed all three IDs in the required order. Automatic grouping returned three singleton groups, so all three predictions used the single-response fallback. This sample exercised the fallback path. It contained no group eligible for soft matching.

At the time of this update, the separate **4,000-row unscored deployment check** had started and its outcome was pending. This candidate had no competition submission or public score.
