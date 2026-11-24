# External training and same-question inference on E24 + E42

The fixed five-fold comparison selected external-first training followed by official fine-tuning and automatic within-question soft matching. It classified **1,042 of 2,160** development responses correctly, **84 more** than the official-only E24 + E42 baseline. All five folds improved. The balanced accuracy gain was **0.038889**, with a 95% paired-question interval of **[0.025000, 0.052778]**. The worst class recall change was −0.033333. The public score of 0.452 belongs to the previously submitted baseline.

| Training data and order | Individual prediction | Automatic soft matching | Kaggle-native groups, soft matching | Known question groups, diagnostic |
|---|---:|---:|---:|---:|
| Official only | 958 · 0.443519 | 1,025 · 0.474537 | 1,025 · 0.474537 | 1,041 · 0.481944 |
| Mixed external and official | 974 · 0.450926 | 1,033 · 0.478241 | 1,034 · 0.478704 | 1,052 · 0.487037 |
| External first, then official | 975 · 0.451389 | **1,042 · 0.482407** | **1,042 · 0.482407** | 1,061 · 0.491204 |

Each cell gives the number correct and balanced accuracy. The original automatic groups and the Kaggle-native groups differ slightly. Moving to the native groups changed 1, 5, and 2 predicted labels under the three training schedules, respectively. Known question IDs are available for the diagnostic column only. All six conditions in the first two result columns were fixed before opening the ten new fold outputs.

## Contributions on the same 120 questions

| Change | Net correct | BA gain | Paired 95% interval |
|---|---:|---:|---:|
| Official only → mixed, individual predictions | +16 | +0.007407 | [−0.003241, +0.017593] |
| Official only → external first, individual predictions | +17 | +0.007870 | [−0.002778, +0.018519] |
| Mixed → external first, individual predictions | +1 | +0.000463 | [−0.011574, +0.012963] |
| Add automatic soft matching to official only | +67 | +0.031019 | [+0.021296, +0.041204] |
| Add automatic soft matching to mixed | +59 | +0.027315 | [+0.016667, +0.037963] |
| Add automatic soft matching to external first | +67 | +0.031019 | [+0.020833, +0.041667] |
| Official only → external first with automatic soft matching | +17 | +0.007870 | [−0.004167, +0.019907] |
| Mixed → external first with automatic soft matching | +9 | +0.004167 | [−0.008333, +0.017130] |
| Automatic → known group IDs with external-first soft matching | +19 | +0.008796 | [+0.003241, +0.015278] |

The fixed automatic matching step produced the larger observed gain. Under external-first training it corrected 100 errors and introduced 33. Against the official-only model already using matching, external-first training corrected 100 and introduced 83. The combined configuration exceeded each component alone on these responses. Its gain over official-only soft matching remains uncertain, as does the advantage of one external training order over the other.

Under the original advancement rule, all three automatic soft conditions passed against the official-only individual baseline. Both new individual-prediction conditions fell short. The rule requires a BA gain of at least 0.010, four non-decreasing folds, and a recall decline strictly below 0.05 in every class. External-first individual prediction had just three non-decreasing folds and a worst recall decline of exactly 0.05. The original trigger for the conditional single-response feature experiment applied to the official-only model. Its known-versus-automatic soft matching gap was 16 predictions, or 0.007407 BA, short of 0.010. The external-first known-group comparison is a separate diagnostic.

As a stability check, 20,000 bootstrap samples of 120 questions repeated the original selection across five candidates. External-first with matching was selected in 42.89% of samples, mixed with matching in 40.01%, and official-only with matching in 16.35%. The remaining 0.75% selected an individual-prediction condition. At least one candidate met the formal filter in 19,360 samples, and the predeclared diagnostic fallback applied in 640. These are conditional frequencies on the reused development set. They do not estimate a probability of winning the hidden test.

## Data and deployment scope

The five folds used 96 official questions for training and 24 for evaluation. Each development question has one response per class. The 30 previous holdout questions were excluded. Both external training schedules used the identical 1,632 verified responses, exposure counts, and update budgets. Added-data comparisons against official-only also include the change in training exposure. Source curation used official global statistics, screening relied on reconstructed official questions, and 1,120 external responses had appeared in earlier baseline development. Training-data results are exploratory.

Automatic matching in the validated physical Kaggle path uses T4-native Qwen3-Embedding-4B features and groups of 2–18. Native cold reloads produced identical archived embeddings. Earlier attempts to reproduce the 3090 grouping under the original cross-device threshold failed in two T4 configurations. The separate native replay used groups frozen before evaluating any of the ten new fold fits and reproduced 1,042 correct for the external-first combination. The official-only native implementation completed a private 4,000-row resource test in 11,344.36 seconds. Among those unlabeled resource inputs, 529 rows were eligible for matching and 3,471 used the single-response fallback. No first choice changed in that resource fixture. Its coverage differs substantially from the development folds, where 2,082 rows were eligible under the original automatic groups. The external-first full-data artifact needs its own technical validation before a hidden score can be attributed to that model.

## First full-data training attempt

After the full-data run loaded the model weights, available working-set memory fell to 3.489 GiB, below the 4 GiB required to begin training. The guard stopped the run before the first optimizer update, so it produced no new model. Peak observed GPU memory use was 13.48 GiB. The maximum measured increase in local storage use was 364 KiB.
