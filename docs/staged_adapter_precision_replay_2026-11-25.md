# Five-fold loading precision comparison for E24 + Qwen3-14B

The original loading method and the method that preserves the saved FP32 classification head each classified **1,042 of 2,160 responses correctly**. Both achieved **0.482407 balanced accuracy**.

Both methods used the same five Qwen adapters trained first on external data and then on official data. The E24 fusion weights, automatic question groups and within-group soft matching were fixed. Each fold evaluated 24 held-out development questions, giving 120 questions across five folds.

The two methods agreed on **2,159 of 2,160** predicted labels. On the remaining response they chose different incorrect labels. The number correct was identical in every fold, and recall was identical for all 18 classes. The balanced accuracy difference was **0.000000**, with a 95% interval of **[0.000000, 0.000000]** from 20,000 paired bootstrap resamples of questions.

The original pipeline was selected on these same development folds. The comparison provides an offline result for the revised loading method. Its hidden-test score remains unmeasured.
