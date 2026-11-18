# Adding a stylometry classifier to E24+E39

The fixed LightGBM branch reduced fivefold balanced accuracy from **0.443056 to 0.438889**. The ensemble lost nine correct predictions across 2,160 responses. It improved in one fold and fell short of the preregistered promotion criteria.

The experiment tested whether an independent implementation of nonlinear stylometry adds information to E24+E39. Its reference point was the stylometry branch in wadigongren’s public ensemble. The author’s **0.684** is a Kaggle public score for that ensemble. The contribution of its own stylometry branch to that score remains unknown. [Author’s scored V1](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline?scriptVersionId=368242626)

## Fixed comparison

The 120 development prompts each had one response from every class. Each fold trained on 96 prompts and predicted the other 24. All 30 previously used holdout prompts stayed outside the experiment. LightGBM 4.6.0 used 90 features computed independently for each text, including word and line lengths, punctuation, formatting and function-word frequencies.

For each response, the classifier’s 18 probabilities were clipped to [0.000001, 1], log-transformed and standardized across classes using the population standard deviation. We added this vector to the archived E24+E39 scores at the fixed weight of 0.25. The standalone classifier reached 0.240278 balanced accuracy.

| Fold | Responses | E24+E39 correct | With stylometry correct | Change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 432 | 188 | 184 | −4 |
| 2 | 432 | 188 | 187 | −1 |
| 3 | 432 | 203 | 201 | −2 |
| 4 | 432 | 186 | 179 | −7 |
| 5 | 432 | 192 | 197 | +5 |
| **Total** | **2,160** | **957** | **948** | **−9** |

The balanced accuracy change was **−0.004167**, with a paired-prompt bootstrap 95% interval of **[−0.012037, +0.003704]** over 20,000 draws (seed 42). Promotion required balanced accuracy of at least 0.453056, a gain of at least 0.01, and at least four nondecreasing folds. The candidate missed all three. It met the class-recall requirement of a drop strictly below 0.05 for every class. The largest drop was for `gpt-6-astra`, from 92/120 to 87/120 (−0.041667). [All 18 classes and five folds](v1_style_lgbm_fivefold_aggregate.json)

## CPU resource measurements

The resource check trained on 1,728 fold-1 training responses and processed 4,000 unlabeled inputs cycled from those texts. It produced 18 finite probabilities per input in a run lasting 91.62 seconds, including offline environment setup. Peak process-tree memory was 257.93 MiB. The full fivefold run took 168.95 seconds and peaked at 272.03 MiB. Both ran locally on four CPU threads.
