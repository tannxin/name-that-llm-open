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

### Soft matching versus pairwise differences

On the same automatic groups, soft matching got 39 more responses right than pairwise differences. It corrected 83 of the pairwise model’s errors and lost 44 of its correct answers. Using known question groups increased the net difference to 49. With automatic groups, soft matching got more answers right on 48 of the 120 development questions, tied on 49 and got fewer right on 23. The comparison uses frozen predictions from the same development set. Whether combining the two methods would improve accuracy remains untested.

| Group information | Correct with pairwise differences | Correct with soft matching | BA difference (soft minus pairwise) | Paired 95% interval | Net correct by fold |
|---|---:|---:|---:|---|---|
| Inferred from responses | 986 | 1,025 | +0.018056 | [+0.007407, +0.028241] | +14, +7, +6, −2, +14 |
| Known question identity | 992 | 1,041 | +0.022685 | [+0.011111, +0.034259] | +14, +7, +14, 0, +14 |

Of the errors made by the original E24 + E42 classifier, the two methods corrected 23 in common with inferred groups. Soft matching alone corrected 70 and pairwise differences alone corrected 20. Among originally correct answers, soft matching alone introduced 24 errors, pairwise differences alone introduced 13, and both introduced 2. With known groups, the corresponding correction counts were 29 shared, 87 soft only and 19 pairwise only. The error counts were 30 soft only, 11 pairwise only and 3 shared.

These exploratory comparisons favor the label-allocation constraint on this split. Pairwise writing comparisons also contributed. The pairwise model gained 24 correct answers with known same-question partners compared with shuffled partners (BA difference 95% interval +0.004167 to +0.018056). With inferred partners it gained 16 (−0.000463 to +0.015278). Intervals use 20,000 paired resamples of questions with seed 42. The comparison uses the archived model outputs and does not change candidate selection.

### Where soft matching corrected predictions

Most of the gain came from inferred groups in which multiple answers had the same baseline top prediction. Among 1,259 such answers, soft matching corrected 84 errors and introduced 24, accounting for 60 of the 67 net additional correct answers (89.6%). Among 823 answers whose top prediction was unique in their group, it corrected nine and introduced two, a net gain of seven. The other 78 retained their baseline predictions. Across eligible groups, the number of unordered answer pairs sharing a top prediction fell from 1,228 to 866. With known question groups, answers with a repeated baseline top prediction accounted for 76 of the 83 net additional correct answers.

Only two inferred groups combined answers to different questions. They contained 37 answers in total. Four went through soft matching, and all four kept their top prediction. Two independent CPU calculations agreed on these counts. This post hoc analysis describes the frozen out-of-fold predictions on 2,160 development answers. The effect of grouping errors on hidden data remains unknown.

### Does soft matching benefit from question membership?

We shuffled group membership 64 times within each fold, keeping every group size, each answer’s eligibility and all E24 + E42 scores fixed. The automatic groups yielded about 56 more correct answers than the average random grouping. Known question groups yielded about 69 more.

| Grouping | Net correct vs baseline | Mean net correct after shuffling (observed range) | BA gain over shuffled mean (paired 95% interval) |
|---|---:|---:|---:|
| Automatic | +67 | +10.97 (−5 to +29) | +0.02594 [+0.01696, +0.03519] |
| Known question identity | +83 | +14.44 (−8 to +33) | +0.03174 [+0.02202, +0.04159] |

The intervals use 20,000 paired resamples of the 120 development questions, conditional on the 64 shuffles. An independent solver reproduced all 140,400 top predictions for each grouping method. This exploratory control supports using question membership on this development set. Deployment still requires the separate cross-device checks below.

### Cross-device deployment check

A private offline T4 run encoded 4,000 unlabeled inputs in 578.68 seconds. On the 2,160 development responses, the minimum cosine similarity to the archived 3090 embeddings was 0.998672, below the required 0.999. Pairwise agreement of the inferred groups had F1 values of 0.987747, 1.000000, 1.000000, 0.994287 and 0.999515 across the five folds. Fold 1 fell below the per-fold requirement of 0.99. It shared 3,305 pairs with the reference, added 51 and lost 31.

With the frozen E24 + E42 scores, all 2,160 soft-matched class predictions agreed across devices. Six responses changed eligibility for group matching. The portability check failed two of its eight requirements. The 3090 reference used BF16 model inference with a batch size of two, while the T4 version used FP16 inference with a batch size of four. These differences were present together, so this comparison cannot isolate their effects.

The 4,000-row timing fixture repeats 1,840 of the 2,160 unique development responses. A CPU replay with the frozen 3090 embeddings and the production vocabulary isolated the effect of these duplicates. Grouping the 2,160 unique responses together admitted 2,077 responses to soft matching. Grouping the 4,000 fixture rows together admitted 540, with 3,453 rows falling back to single-response prediction because their groups exceeded 18 rows and seven falling back as singletons. The actual T4 run admitted 529 of the same 4,000 rows. Hidden-test coverage depends on how the real questions and responses are distributed.

The first T4 probe also padded each batch of four to its longest text. Relative to the archived two-text batches, 1,076 of 2,160 texts received additional left padding. Effective token IDs agreed for all 2,160. A second private probe kept those token IDs, restored the original two-text batches and left-padding shapes, and applied the same acceptance thresholds.

The full pipeline completed a private, offline rehearsal on two T4s in 11,344.36 seconds (3 h 9 min). The lowest sampled free space on the working disk was 13.24 GiB. Preparation, E24 scoring, E42 scoring, 4B encoding in batches of four, group matching and independent readback all completed. The readback verified 18 finite scores per row from each scoring branch and reproduced all 4,000 Parquet predictions in the expected ID order. The 14B model peaked at 7.07 and 6.26 GiB of reserved GPU memory on the two cards. The 4B encoder peaked at 8.25 GiB on one card. [Kaggle resource run, version 372165904](https://www.kaggle.com/code/tanxin357/ntl-e42-soft-groups-4000-resource?scriptVersionId=372165904)

The model inferred 213 groups. Soft matching applied to 529 rows, while 3,471 rows used the original single-response predictions. No top prediction changed on this repeated, unlabeled fixture. The batch-four encoder passed six of eight cross-device checks. Its two outstanding discrepancies still prevent a competition submission using that configuration.

The separate official three-row preview returned a 1,761-byte Parquet file, retrieved directly from the saved version. [Kaggle sample preview, version 372153035](https://www.kaggle.com/code/tanxin357/ntl-e42-automatic-soft-grouping?scriptVersionId=372153035)

In the batch-two run, the T4 encoded all 4,000 rows in 442.83 seconds including model loading. On the 2,160 unique development responses, the minimum cosine to the 3090 reference was **0.998656**, below 0.999. Group pair F1 across the five folds was 0.987747, 0.993452, 1.000000, 0.994287 and 0.999515. Fold 1 again fell below 0.99. The two devices agreed on 2,159 of 2,160 soft-matched predictions, and seven responses changed eligibility for matching. Batch two passed the same six of eight checks as batch four. Matching batch partners and padding did not resolve the cross-device discrepancy. The two devices also used different inference precision, leaving the cause unresolved. [Kaggle batch-two output, version 372271781](https://www.kaggle.com/code/tanxin357/ntl-q4b-group-portability-v1/output?scriptVersionId=372271781)

On the frozen clustering pipeline, the T4 and RTX 3090 disagreed on whether 166 pairs of responses belonged to the same group. For only three of those pairs did the direct distance cross the 0.60 threshold between devices. Replaying the RTX 3090 merge order with T4 distances reproduced the 3090 assignment for all 166 pairs. This isolates the effect of merge order in the replay. The experiment leaves the underlying hardware or numerical cause unresolved, and the T4 configuration still passes six of eight cross-device checks.

Two fresh processes on the same Kaggle T4 each encoded the 2,160 development responses in batches of four, with each fold packed separately. Both produced byte-identical arrays of 2,160 × 2,560 values, identical inferred links in all five folds and identical soft-matched choices for all responses. The private offline run took 898.9 seconds. Independent verification used the two original arrays, their manifests and the aggregate report. [Kaggle same-device output, version 372342874](https://www.kaggle.com/code/tanxin357/ntl-q4b-group-portability-v1/output?scriptVersionId=372342874)

The host scorer recorded the first process’s groups and choices before reading the development labels. Applying those groups to the frozen E24 + E42 scores raised balanced accuracy from **0.443519 to 0.474537**. Soft matching corrected 93 answers and lost 26, for a net gain of 67. All five folds improved, with gains of 19, 14, 13, 7 and 14 answers. The 95% interval for the increase was +0.021296 to +0.041204 under 20,000 paired resamples of questions. This is an exploratory development result on the original 120 questions. The cross-device requirements in the batch-two and batch-four experiments still govern competition eligibility.

For the earlier batch-two cross-device run (version 372271781), the single changed class prediction occurred in fold 2. With 1,025 archived soft-matched decisions correct, changing one decision bounds the result for that run at **1,024–1,026 correct out of 2,160**, or 66–68 above baseline. That run reported agreement counts and did not return the individual T4 predictions for scoring.

A CPU reconstruction of the original RTX 3090 batches and Kaggle T4 slot A recovered identical unpadded token sequences and the same last valid token for all 2,160 responses. In the archived normalized embeddings, 26,606 of 5,529,600 coordinates had opposite nonzero signs, with at least one in every response. Positive L2 scaling and the final storage cast preserve the sign of nonzero values, so the differences already existed before normalization. These archives cannot isolate the effects of forward precision, batch layout and device kernels.

### BF16 operator diagnostic on the T4

A separate private run kept the 4B weights and inter-layer activations in BF16. It evaluated the linear projections and math attention in FP32 and rounded their outputs back to BF16. The T4 encoded all 4,000 fixture rows in 2,612.01 seconds, including model loading. The preceding FP16 run with the same two-row batches took 442.83 seconds. Both used the same frozen 3090 references, inputs and numerical thresholds.

Six of eight checks passed. Minimum embedding cosine was 0.998385 (required 0.999), and fold 1 group-pair F1 was 0.983046 (required 0.99). Across the 3,388 inferred fold 1 pairs, 3,305 matched the 3,336 reference pairs. The other four folds met the F1 threshold. The final soft-matched class agreed on 2,159 of 2,160 development rows. The changed decision was in fold 2. Retaining BF16 at operator boundaries did not restore cross-device agreement under these conditions. This diagnostic measured agreement with an archived run and did not measure hidden-test accuracy.

The platform produced a 6,560-byte report (SHA-256 `7844dc96053ba55db4d12bac058406f54e10d92851639c0360d37a48d638a0ba`). The host independently recomputed all eight checks and reproduced the report bytes from the second copy printed in the version log.

## External data for the four-view SVM

Open Pool v17 and NLM Matched Clean v3 contained 226,191 and 24,443 rows. After checking model versions, source prompts and duplicates, 7,008 responses were eligible. Of those, 4,260 had raw request/response evidence and 2,748 were linked to a direct Google Vertex call through the generator's recorded `modelVersion`. The fixed sample contained 1,632 responses over 547 questions and 17 classes, with 96 distinct questions per represented class. Earlier E24/v9 training had used 1,120 of these responses, leaving 512 previously unused.

Before the 96-question cap, 1,100 of 1,129 eligible questions had multiple answers. These counts cover responses that passed source verification and deduplication. Each question had at most one answer per model class, and none covered all 18 classes. Enforcing one answer per question and class removed zero responses. The cap removed 5,376 responses, leaving 312 questions with multiple answers in the final training sample. This pattern describes how the two external sources sampled answers. Whether a model can appear twice in a hidden-test question group remains unknown.

| Comparison | Reference BA | Expanded SVM BA | Net correct | Paired 95% interval for BA change |
|---|---:|---:|---:|---:|
| Standalone four-view SVM, official-only vs expanded training | 0.358333 | 0.366204 | +17 | [−0.010648, +0.026389] |
| Same E24 + E42 fusion, official-only vs expanded SVM | 0.446759 | 0.447222 | +1 | [−0.005556, +0.006481] |
| E24 + E42 alone vs adding the expanded SVM | 0.443519 | 0.447222 | +8 | [−0.001852, +0.009722] |

Each fusion adds 0.25 times the row-standardized SVM scores to the baseline. The matched fusion comparison corrected 22 cases and lost 21. Restricting the external source evidence to raw HTTP records also gained one correct prediction in fusion. Neither data comparison passed the advancement rule. These measurements apply to the SVM training recipe. The comparison of Qwen training schedules is separate.

We also compared the 547 selected external questions with all 150 available reconstructions of the official questions (82,050 pairs). For a blinded review, we took the six highest word unigram/bigram similarities and the six highest character four-gram similarities. The two lists had no pair in common. Two AI reviewers independently judged that each of the 12 pairs asked for a different task. This check covers the closest wording in this sample. Substantial paraphrases elsewhere and errors in the reconstructed questions remain possible.

The verified external sample contained zero responses from `qwen3.8-max`. On its 120 development responses, the expanded standalone SVM identified 49 correctly, down from 60. Across all classes it predicted that label 102 times, down from 146. The other 17 classes gained 28 correct predictions in total, giving the overall net gain of 17. In the fixed E24 + E42 fusion the class retained 64 correct predictions, with one correction and one newly introduced error. The other 17 classes accounted for the single net gain in fusion. These counts describe this training recipe and cannot isolate the effect of missing external examples for one class.

### How the SVM changes carried through to the ensemble

On the 2,160 development responses, adding external data changed the standalone SVM’s top prediction on 869 responses. Replacing the official-only SVM with the expanded SVM in the fixed ensemble changed 86 predictions. The table pairs the two comparisons response by response. Correctness stays the same when both predictions are correct or when both are wrong.

| Standalone SVM after adding external data | Ensemble corrected | Ensemble newly wrong | Ensemble correctness stayed the same |
|---|---:|---:|---:|
| Corrected | 7 | 1 | 202 |
| Newly wrong | 0 | 3 | 190 |
| Correctness stayed the same | 15 | 17 | 1,725 |

Of the 210 responses the expanded SVM corrected on its own, both ensembles already got 91 right and both missed 111. Across the 869 responses where the SVM’s top prediction changed, the ensemble gained nine and lost nine. Across the other 1,291 responses, the SVM kept its top prediction, but its 18 class scores could still change. The ensemble changed 39 predictions there, correcting 13, losing 12 and switching between two wrong classes on 14. This accounts for the ensemble’s net gain of one on the fixed folds. Relative to E24 + E42 alone, adding the official-only SVM changed 108 predictions. Adding the expanded SVM changed 100.

The catalogs used information from the full official corpus during curation, and upstream generation used official length statistics. The 150 official questions were screened using reconstructed question text, which can miss substantial paraphrases. Official answer hashes were also screened for exact duplicates without using labels. The external-data effect is therefore exploratory. The E24 baseline itself includes an external source informed by official length statistics.

## Qwen training status

The first two Qwen3-14B-Base runs tested mixed training on fold 1 and external training followed by official fine-tuning on fold 2. Both last recorded 900 of 1,260 planned optimizer steps. Each GPU guard detected an additional compute process and stopped its own run. Across the two five-fold schedules, 0 of 10 runs completed. The two interrupted fits would have to start from scratch because they saved no resumable checkpoint. Validation scores for both training schedules remain unknown.

On the authorized restart, fold 1 mixed training began after 601.8 seconds of continuous GPU availability. An additional Python process appeared about ten minutes into training, and the guard stopped the fit. The run produced no checkpoint. At 14:49 UTC on 21 November, fold 2 in the external-then-official schedule had reached 840 of 1,260 steps and was still running. No fold score was used to change either schedule.

At 15:45 UTC on November 21, fold 2 completed external training followed by official fine-tuning. The mixed run on that fold then started on the same 3090. On the other 3090, the guard stopped the fold 1 external-then-official worker during setup when another process appeared. The worker had completed zero optimizer updates. The five fits that had yet to enter training were queued behind the original four on the available 3090. At 16:44 UTC, one of ten fits had completed. The mixed run on fold 1 still required a further restart decision.

At 17:22 UTC on 21 November, the storage guard stopped mixed training on fold 2 after local disk use grew past its 2 GiB limit. The largest observed increase around the stop was in another workload’s local directory. The available measurements do not attribute every byte of the increase. The queued continuation ended before it started a fit.

Across the ten planned fits, one completed, two stopped during training without a resumable checkpoint, and seven had not entered training. The completed fit was fold 2 in the external-then-official schedule. A comparison of the two training schedules still requires complete predictions from both across all five folds.

At 18:18 UTC on 21 November, the controller for the seven previously untrained fits launched after a continuous ten-minute GPU and disk check. At dispatch, one of the ten fits had completed. The two previously interrupted mixed-training fits were outside this queue.

At 20:52 UTC the controller stopped fold 1 external-then-official training after its disk-usage command exited with an error. The last log entry recorded 360 of 1,260 updates. The command’s diagnostic text was unavailable. At 21:19 UTC a new queue began its resource checks for the six untouched fits covering both training schedules on folds 3–5. Its CPU preflight verified the model and all 1,632 selected external responses. The checkpoint writer targets shared storage every 60 updates. At 21:32 UTC the first fit was waiting for the GPU and the monitored disk writer to stay idle together. One fit had completed and three had stopped without a resumable checkpoint. Neither five-fold training comparison was available.

The storage guard stopped the six-fit queue at 21:50 UTC, before any model training. Independent inspection of all six output locations confirmed zero worker starts and zero checkpoints. Two minutes later, local disk usage was 2.23 GiB above the recorded baseline, exceeding the 2 GiB growth limit. Local free space was still about 219 GiB. A read-only observer subsequently began checking for a stable training window. The training comparison remains at one completed fit out of ten.

## 22 November: 14B verification on a dedicated RTX 4090 D

The frozen fold-1 adapter reproduced the RTX 3090 top prediction on all three reference texts. The largest absolute difference across the 54 logits was 0.008789, within the prespecified tolerance of 0.05. After that process exited, a fresh process ran a 256-token synthetic training step with NF4, FP16 and AdamW. The update succeeded and peak reserved GPU memory was 13.67 GiB against a 22 GiB limit. Both processes exited normally. Across both checks, the largest measured increase on the writable memory-backed volumes was 8 KiB.

The six original mixed and external-then-official fits for folds 3-5 were assigned to this device, with each pair on the same physical GPU. The completed fold-2 external-then-official predictions were archived against their original digest. Fold-level accuracy remains sealed until all ten fits are available.

At 08:28 UTC on 22 November, the memory and Pod identity guard stopped the fold-3 mixed-training run. The last sealed checkpoint was at group 120 of 1,260, with 115 successful updates and five skipped updates. A separate CPU process read both checkpoints and verified their hash chain, all 561 trainable tensors, all 1,122 AdamW moment tensors, and the scheduler and data positions. The Pod remained Ready with zero restarts. Across 479 recorded samples, monitored local storage grew by at most 104 KiB. The final sample that triggered the stop was not recorded. One of ten fits has completed, and five fits in the dedicated queue have yet to start.

At 09:30 UTC the fold-3 mixed fit resumed on the same RTX 4090 D. Optimizer group 121 advanced the training cursor from 960 to 968 examples and raised the successful update count from 115 to 116, with five skipped updates. At 09:36 UTC a separate process streamed and hashed the new 772,517,828-byte checkpoint written after group 180. By that readback, resource samples had recorded a maximum increase of 88 KiB in monitored Pod-local storage. One of the ten planned fits had completed.

## 22 November: fold 3 mixed training and validation memory

At 11:30 UTC the fold-3 mixed run stopped after training and validation. It processed all 10,080 example presentations and all 1,260 optimizer groups, with 1,255 successful updates and five skipped updates. The final checkpoint recorded 7,580.29 seconds of training and 13.742 GiB peak reserved GPU memory. An independent CPU comparison matched all 561 trainable tensors in the saved adapter to the final checkpoint, covering 64,317,440 float32 values.

The worker computed a finite 18-class score vector for each of the 432 validation answers, then checked the original limit of strictly less than 22 GiB for each phase. The recorded training peak and the failed check establish that validation reached at least 22 GiB. Its exact PyTorch peak was not persisted. The process exited before writing fold predictions. Monitored Pod-local storage grew by at most 160 KiB and the sampled OOM-kill count stayed at zero. One of the ten comparisons has a complete fold output. The other five fits in the dedicated queue had not started at the time of the stop.

## Fold 3 mixed validation in a separate process

On 22 November, a fresh process on the RTX 4090 D loaded the saved fold-3 mixed-training adapter and produced predictions for all 432 held-out answers in their original order. Training peaked at 13.74 GiB of reserved GPU memory. Validation peaked at 13.31 GiB, and inference took 75.54 seconds. Monitored Pod-local storage grew by at most 80 KiB during this validation run.

The blind audit accepted this output, bringing the total to two of ten across the paired five-fold schedules. The archive retains the earlier failure records. Fold scores stay sealed until all ten outputs pass the audit.

## Fold 3 external training followed by official fine-tuning

The training and validation processes both exited successfully on 22 November. On the dedicated 48 GiB RTX 4090 D, training took 7,672.77 seconds and peaked at 13.74 GiB of reserved GPU memory. Fresh-process validation took 76.82 seconds, peaked at 13.31 GiB, and produced predictions for all 432 held-out answers. The 1,260 scheduled optimizer groups yielded 1,255 successful updates and five skips. Monitored Pod-local storage grew by at most 216 KiB.

Two independently hashed checkpoints before and after the switch matched the prescribed data order. At group 600, the cursor covered 4,800 external presentations. At group 660, it covered all 4,896 external presentations and the first 384 official presentations. The optimizer and learning-rate schedule continued across that boundary.

Three of the ten planned fits have now produced fold predictions. The host checked the new fold’s shared copies, its chain of 21 checkpoint receipts and the prediction file hash. Confirmation against the dedicated Pod’s private originals is pending, so two fits have completed independent acceptance. The fold predictions remain sealed.

### Training exposure and learning rate by data source

Both expanded schedules prescribe three presentations of each of the same 1,728 official and 1,632 external training answers per fold. The official-only runs used the same official answers three times. The five-fold expanded totals come from the locked inputs and the 256-token truncation rule. Both completed fold-3 training records match the calculated 1,451,586 tokens for that fold.

| Training budget | Official only | Mixed or external-first |
|---|---:|---:|
| Official presentations per fold | 5,184 | 5,184 |
| External presentations per fold | 0 | 4,896 |
| Planned optimizer updates per fold | 648 | 1,260 |
| Linear warmup steps per fold | 20 | 38 |
| Training tokens across five folds, per schedule | 3,730,596 | 7,230,576 |

The external rows add 699,996 tokens per fold. Comparing either expanded arm with official-only training therefore measures the joint effect of new data and a 93.82% larger total token budget across five folds. Comparing the two expanded arms holds that budget and both within-source example orders fixed.

The staged schedule carries AdamW and its learning-rate scheduler across the switch to official data. In fold 3, the first official group was group 613, with a learning rate of 0.000106874. Across the 5,184 official presentations, the average group learning rate was 0.000053928 for staged training and between 0.000099607 and 0.000100618 for mixed training. Each skipped group contributes zero to this average. The bounds cover all placements of the five mixed-training skips compatible with the observed count at group 120. Thus official examples in the staged arm received about 54% of the average learning rate they saw in mixed training. This characterizes how the order interacts with the decay schedule. Accuracy is assessed separately on the paired five-fold predictions.

### Fold 4 paired training and validation

Both training orders finished on the same RTX 4090 D. Each processed 1,440,117 tokens, with 1,255 successful optimizer updates and five skipped updates. Both validation runs produced the 432 held-out IDs in the order independently reconstructed from the official split. At group 600 the mixed run had seen 2,480 official and 2,320 external presentations. The staged run had seen 4,800 external presentations at that point. Both finished with 5,184 official and 4,896 external presentations.

| Fold 4 measurement | Mixed | External first, then official |
|---|---:|---:|
| Recorded training time | 7,749.79 s | 7,733.09 s |
| Validation inference time | 72.09 s | 75.82 s |
| Peak GPU memory reserved during training | 13.744 GiB | 13.740 GiB |
| Peak GPU memory reserved during validation | 13.309 GiB | 13.309 GiB |
| Maximum monitored Pod-local storage growth | 476 KiB | 348 KiB |

Independent CPU readback compared each exported adapter with its own final checkpoint. In each arm all 64,317,440 FP32 values in 561 tensors matched byte for byte, including the classification head.

### Fold 5 paired training and validation

Both arms ran on the same RTX 4090 D. Each processed 1,446,840 training tokens, with 1,255 successful optimizer updates and five skipped updates. The mixed and staged runs each matched their respective frozen example order. Their validation files contain the same 432 unique IDs in the official order from 24 held-out development questions and all 18 score columns.

| Fold 5 measurement | Mixed | External first, then official |
|---|---:|---:|
| Recorded training time | 7,718.36 s | 7,705.38 s |
| Validation inference time | 76.14 s | 74.51 s |
| Peak GPU memory reserved during training | 13.742 GiB | 13.742 GiB |
| Peak GPU memory reserved during validation | 13.309 GiB | 13.309 GiB |
| Maximum monitored Pod-local storage growth | 600 KiB | 728 KiB |

The five-run queue recorded successful exits for both fold 5 training workers and both validation workers. Independent readback matched the recorded hashes of the two exported adapters and prediction files. Each arm has a continuous chain of 21 shared checkpoint receipts through its last update.

The independent host audit has accepted seven of ten runs. It verified 138 original records from the dedicated training container, including 105 checkpoint receipts across five runs, and matched each of their five prediction files to its recorded SHA-256. Both training orders are registered for folds 3–5, along with external-first training for fold 2. Both fold 1 runs and the fold 2 mixed run are outstanding. The comparison will use all ten runs before opening the validation scores.
