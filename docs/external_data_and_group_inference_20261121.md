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

The 4,000-row timing fixture repeats 1,840 of the 2,160 unique development responses. A CPU replay with the frozen 3090 embeddings and the production vocabulary isolated the effect of these duplicates. Grouping the 2,160 unique responses together admitted 2,077 responses to soft matching. Grouping the 4,000 fixture rows together admitted 540, with 3,453 rows falling back to single-response prediction because their groups exceeded 18 rows and seven falling back as singletons. The actual T4 run admitted 529 of the same 4,000 rows. Hidden-test coverage depends on how the real questions and responses are distributed.

The first T4 probe also padded each batch of four to its longest text. Relative to the archived two-text batches, 1,076 of 2,160 texts received additional left padding. Effective token IDs agreed for all 2,160. A second private probe kept those token IDs, restored the original two-text batches and left-padding shapes, and applied the same acceptance thresholds.

The full pipeline completed a private, offline rehearsal on two T4s in 11,344.36 seconds (3 h 9 min). The lowest sampled free space on the working disk was 13.24 GiB. Preparation, E24 scoring, E42 scoring, 4B encoding in batches of four, group matching and independent readback all completed. The readback verified 18 finite scores per row from each scoring branch and reproduced all 4,000 Parquet predictions in the expected ID order. The 14B model peaked at 7.07 and 6.26 GiB of reserved GPU memory on the two cards. The 4B encoder peaked at 8.25 GiB on one card. [Kaggle resource run, version 372165904](https://www.kaggle.com/code/tanxin357/ntl-e42-soft-groups-4000-resource?scriptVersionId=372165904)

The model inferred 213 groups. Soft matching applied to 529 rows, while 3,471 rows used the original single-response predictions. No top prediction changed on this repeated, unlabeled fixture. The batch-four encoder passed six of eight cross-device checks. Its two outstanding discrepancies still prevent a competition submission using that configuration.

The separate official three-row preview returned a 1,761-byte Parquet file, retrieved directly from the saved version. [Kaggle sample preview, version 372153035](https://www.kaggle.com/code/tanxin357/ntl-e42-automatic-soft-grouping?scriptVersionId=372153035)

In the batch-two run, the T4 encoded all 4,000 rows in 442.83 seconds including model loading. On the 2,160 unique development responses, the minimum cosine to the 3090 reference was **0.998656**, below 0.999. Group pair F1 across the five folds was 0.987747, 0.993452, 1.000000, 0.994287 and 0.999515. Fold 1 again fell below 0.99. The two devices agreed on 2,159 of 2,160 soft-matched predictions, and seven responses changed eligibility for matching. Batch two passed the same six of eight checks as batch four. Matching batch partners and padding did not resolve the cross-device discrepancy. The two devices also used different inference precision, leaving the cause unresolved. [Kaggle batch-two output, version 372271781](https://www.kaggle.com/code/tanxin357/ntl-q4b-group-portability-v1/output?scriptVersionId=372271781)

A separate same-T4 protocol compares two independent model loads, each encoding the same 2,160 development texts in fold-packed batches of four. It tests repeatability on T4. The 4,000-row end-to-end rehearsal above used batches of four. The batch-two diagnostic measured the encoder and grouping steps and has no competition score.

The single changed prediction occurred in fold 2. An independent check matched the frozen E24 + E42 scores, row order and fold assignments to the archived comparison. The baseline had 958 correct predictions and the archived soft assignment had 1,025. Changing at most one of those decisions bounds the T4 result with these same input scores at **1,024–1,026 correct out of 2,160**, a gain of 66–68. The lower bounds on net gains across the five folds were 19, 13, 13, 7 and 14. This bound applies to this development set with fixed input scores. The probe did not score the hidden test or the new training schedules. Both T4 configurations missed the embedding and grouping tolerances.

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

At 15:45 UTC on November 21, fold 2 completed external training followed by official fine-tuning. The mixed run on that fold then started on the same 3090. On the other 3090, the guard stopped the fold 1 external-then-official worker during setup when another process appeared. The worker had completed zero optimizer updates. The five fits that had yet to enter training were queued behind the original four on the available 3090. At 16:44 UTC, one of ten fits had completed. The mixed run on fold 1 still required a further restart decision.

At 17:22 UTC on 21 November, the storage guard stopped mixed training on fold 2 after local disk use grew past its 2 GiB limit. The largest observed increase around the stop was in another workload’s local directory. The available measurements do not attribute every byte of the increase. The queued continuation ended before it started a fit.

Across the ten planned fits, one completed, two stopped during training without a resumable checkpoint, and seven had not entered training. The completed fit was fold 2 in the external-then-official schedule. A comparison of the two training schedules still requires complete predictions from both across all five folds.

At 18:18 UTC on 21 November, the controller for the seven previously untrained fits launched after a continuous ten-minute GPU and disk check. At dispatch, one of the ten fits had completed. The two previously interrupted mixed-training fits were outside this queue.

At 20:52 UTC the controller stopped fold 1 external-then-official training after its disk-usage command exited with an error. The last log entry recorded 360 of 1,260 updates. The command’s diagnostic text was unavailable. At 21:19 UTC a new queue began its resource checks for the six untouched fits covering both training schedules on folds 3–5. Its CPU preflight verified the model and all 1,632 selected external responses. The checkpoint writer targets shared storage every 60 updates. At 21:32 UTC the first fit was waiting for the GPU and the monitored disk writer to stay idle together. One fit had completed and three had stopped without a resumable checkpoint. Neither five-fold training comparison was available.
