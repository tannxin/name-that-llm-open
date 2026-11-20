# Which components of the 0.684 notebook transfer to our ensemble?

The Gaussian group readout showed the clearest benefit on a fixed representation. With the same frozen Qwen3-Embedding-4B features, it raised five-fold balanced accuracy (BA) from 0.207407 for logistic regression to 0.300000. It gained 200 correct predictions across 2,160 responses, improved all five folds and all 18 class recalls, and passed the prespecified head comparison. The paired 95% interval for the gain was [+0.069907, +0.115741]. The author's scored notebook provides the reference implementation. [Original Kaggle V1](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline?scriptVersionId=368242626)

## Effect on the existing ensembles

| Component transferred | Fixed baseline | Baseline BA | With component | Net correct | Non-decreasing folds | 95% interval for BA change |
|---|---|---:|---:|---:|---:|---|
| Gaussian group readout | E24+E42 | 0.443519 | 0.452778 | +20 | 5/5 | [+0.004167, +0.014815] |
| Four-view linear SVC | E24+E42 | 0.443519 | 0.446759 | +7 | 5/5 | [−0.003241, +0.009722] |
| Group centering of StyleGrams | E24+E39 | 0.443056 | 0.444907 | +4 | 3/5 | [−0.004167, +0.007407] |
| 90-feature style LightGBM | E24+E39 | 0.443056 | 0.438889 | −9 | 1/5 | [−0.012037, +0.003704] |

The Gaussian branch corrected 30 ensemble errors and introduced 10. Its BA gain of 0.009259 fell short of the 0.010 promotion threshold. Replacing a logistic head already in the ensemble with the Gaussian head gained 14 correct predictions. The four-view SVC had a larger effect inside the three refitted sparse classifiers, where adding it back gained 40 correct predictions (95% interval for BA change [+0.003241, +0.033796]).

## What the experiment measures

Every new supervised component trained on 96 development prompts per fold and was evaluated on the other 24. The 30 previously used holdout prompts were excluded. Intervals use 20,000 paired bootstrap samples of the 120 development prompts. The StyleGrams comparison adds only the difference between centered and plain scores for eligible groups. The LightGBM experiment uses an independently implemented feature set.

The Gaussian experiment uses last-token embeddings for classification and grouping. The original 4B branch uses a decoder layer mean for classification and GLM features for grouping. Inferred groups depended on the unlabeled batch and covered 2,082 development responses. Repeated use of these development prompts and the E24 external data source's use of aggregate official length statistics limit generalization. The author's 0.684 leaderboard score is a separate observation. Its per-component contributions remain unknown.

## Aggregate source checksums

| Report | SHA-256 |
|---|---|
| Gaussian transfer | `9c7290404492b5cd976a45b19cd58c35ebe924d969c4d13625552826dbc33e47` |
| Sparse classifier refits | `b2027506ae932c819c24e8a3bd338f12175ec26be1760e9c7338d743bf249267` |
| StyleGrams centering | `b7bf4a8f153fabde48dc5b098172e4d43fb40fb47b1a1d50ce737c67f35f04e7` |
| Style LightGBM | `73234d433c7a76a6b29e32fe928254857dab3b10fcbebed5bbc380e6ec82dcdc` |
