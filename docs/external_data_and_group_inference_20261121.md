# External training data and same-question inference

Automatic question grouping with soft label matching raised five-fold balanced accuracy (BA) from **0.443519 to 0.474537** on 2,160 development responses. It corrected 93 previously wrong predictions and lost 26 correct ones. At the same fusion weight, expanding the four-view SVM training set gained **one correct prediction** over the official-only SVM. Both comparisons use E24 + E42 as the underlying classifier.

The previously submitted E24 + E42 baseline has a public score of **0.452**. An earlier, separate reproduction of a third-party system scored **0.684**. The results below are development scores. As of this report on November 21, 2026, these comparisons have no verified public score.

## Shared evaluation

Five folds partition 120 official development questions. Each fold trains on 96 questions and evaluates the other 24 across 18 model classes. The 30 previously used holdout questions remain outside fitting and evaluation. Each comparison uses 20,000 paired bootstrap resamples of questions. The predefined advancement rule requires at least +0.010 BA, at least four non-decreasing folds, and a recall decline strictly below 0.05 for every class.

## Same-question inference

| Inference method | Automatic groups BA | Known question groups BA |
|---|---:|---:|
| Original Gaussian readout added to E24 + E42 | 0.452778 | 0.455093 |
| Pairwise-difference readout added to E24 + E42 | 0.456481 | 0.459259 |
| Soft label matching applied directly to E24 + E42 | **0.474537** | 0.481944 |

Automatic groups came from fold-trained TF-IDF and frozen Qwen3-Embedding-4B features on unlabeled responses. Known question groups provide a diagnostic comparison. Soft matching penalizes excessive predicted occupancy of the same label in groups of 2–18. Rows outside that range retain their individual scores.

Against E24 + E42 alone, automatic soft matching gained 67 correct predictions. Net gains by fold were +19, +14, +13, +7 and +14. The paired 95% interval for the BA gain was [+0.021296, +0.041204], and the largest class recall decline was 0.016667. Pairwise differences also passed the advancement rule against E24 + E42. Their advantage over the original automatic Gaussian readout was smaller at eight correct predictions, with a paired BA interval of [−0.002315, +0.009722].

Using known groups recovered 16 more correct predictions than automatic soft matching, a BA difference of +0.007407. Automatic groups of eligible size covered 2,082 of 2,160 responses. Development questions contain exactly one response per class. The usefulness of this constraint on hidden data depends on its group composition.

### Cross-device deployment check

A private offline T4 run encoded 4,000 unlabeled inputs in 578.68 seconds. On the 2,160 development responses, the minimum cosine similarity to the archived 3090 embeddings was 0.998672, below the required 0.999. Pairwise agreement of the inferred groups had F1 values of 0.987747, 1.000000, 1.000000, 0.994287 and 0.999515 across the five folds. Fold 1 fell below the per-fold requirement of 0.99. It shared 3,305 pairs with the reference, added 51 and lost 31.

With the frozen E24 + E42 scores, all 2,160 soft-matched class predictions agreed across devices. Six responses changed eligibility for group matching. The portability check failed two of its eight requirements. The 3090 reference used BF16 model inference with a batch size of two, while the T4 version used FP16 inference with a batch size of four. These differences were present together, so this comparison cannot isolate their effects.

## External data for the four-view SVM

Open Pool v17 and NLM Matched Clean v3 contained 226,191 and 24,443 rows. After checking model versions, source prompts and duplicates, 7,008 responses were eligible. Of those, 4,260 had raw request/response evidence and 2,748 were linked to a direct Google Vertex call through the generator's recorded `modelVersion`. The fixed sample contained 1,632 responses over 547 questions and 17 classes, with 96 distinct questions per represented class. Earlier E24/v9 training had used 1,120 of these responses, leaving 512 previously unused.

| Comparison | Reference BA | Expanded SVM BA | Net correct | Paired 95% interval for BA change |
|---|---:|---:|---:|---:|
| Standalone four-view SVM, official-only vs expanded training | 0.358333 | 0.366204 | +17 | [−0.010648, +0.026389] |
| Same E24 + E42 fusion, official-only vs expanded SVM | 0.446759 | 0.447222 | +1 | [−0.005556, +0.006481] |
| E24 + E42 alone vs adding the expanded SVM | 0.443519 | 0.447222 | +8 | [−0.001852, +0.009722] |

Each fusion adds 0.25 times the row-standardized SVM scores to the baseline. The matched fusion comparison corrected 22 cases and lost 21. Restricting the external source evidence to raw HTTP records also gained one correct prediction in fusion. Neither data comparison passed the advancement rule. These measurements apply to the SVM training recipe. The comparison of Qwen training schedules is separate.

The catalogs used information from the full official corpus during curation, and upstream generation used official length statistics. The 150 official questions were screened using reconstructed question text, which can miss substantial paraphrases. Official answer hashes were also screened for exact duplicates without using labels. The external-data effect is therefore exploratory. The E24 baseline itself includes an external source informed by official length statistics.

## Qwen training status

The first two Qwen3-14B-Base runs tested mixed training on fold 1 and external training followed by official fine-tuning on fold 2. Both last recorded 900 of 1,260 planned optimizer steps. Each GPU guard detected an additional compute process and stopped its own run. Across the two five-fold schedules, 0 of 10 runs completed. The two interrupted fits would have to start from scratch because they saved no resumable checkpoint. Validation scores for both training schedules remain unknown.

On the authorized restart, fold 1 mixed training began after 601.8 seconds of continuous GPU availability. An additional Python process appeared about ten minutes into training, and the guard stopped the fit. The run produced no checkpoint. At 14:49 UTC on 21 November, fold 2 in the external-then-official schedule had reached 840 of 1,260 steps and was still running. No fold score was used to change either schedule.
