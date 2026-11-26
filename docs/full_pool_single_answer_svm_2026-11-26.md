# Full-catalog external training for single-answer classification

On the fixed fivefold split, a four-view linear SVM trained first on both external catalogs and then on official answers classified **957 of 2,160** development answers correctly (balanced accuracy **0.443056**). The same solver, trained in the same order with the prior 1,632-row external pool, classified 675 correctly. The gain was **282 answers**, or 0.130556 balanced accuracy (95% question-paired bootstrap interval 0.112963–0.148160). The five folds gained 65, 61, 54, 58 and 44 answers.

| External stage | Official fine-tuning | Correct / 2,160 | Balanced accuracy |
|---|---|---:|---:|
| None | Three epochs | 630 | 0.291667 |
| Prior 1,632-row pool | Three epochs | 675 | 0.312500 |
| 173,501 rows with checkable question provenance passing the topic screen | Three epochs | 936 | 0.433333 |
| All 250,634 catalog rows, duplicates retained | Three epochs | 957 | 0.443056 |

## Contribution alongside two frozen single-answer baselines

Each addition uses the previously fixed coefficient of 0.25 on row-standardized SVM scores. Inputs, official folds and baseline weights are fixed. All predictions use a single answer, with no group information.

| Baseline | Alone | With full-catalog SVM | Change | 95% question-paired interval for BA change |
|---|---:|---:|---:|---:|
| E24 + official-only Qwen3-14B | 958 (0.443519) | 978 (0.452778) | +20 | +0.003241 to +0.015741 |
| E24 + prior small-pool external-first Qwen3-14B, precision replay B | 976 (0.451852) | **999 (0.462500)** | **+23** | **+0.003241 to +0.018519** |

For the second baseline, the five folds changed by +7, +7, +5, −2 and +6. The largest class recall decline was 0.033333. The question-screened external SVM added to that same baseline classified 1,001 correctly. The official-only SVM added to it classified 978 correctly. The 14B model in this table used the prior small training pool. These measurements provide no result for a new full-catalog 14B model or a competition submission.

## Rows, weights and interpretation

Open Pool v17 contributed 226,191 rows and NLM Matched Clean v3 contributed 24,443. All 250,634 physical rows received positive loss weights, including 17,252 duplicated occurrences across the catalogs. An exact normalized-text check found zero collisions with official answers. Previously verified upstream records, exact returned model IDs, source documentation and catalog-only labels received different reliability weights. The final manifest recovered upstream evidence for 4,219 matched-only rows initially assigned catalog-only confidence. A clipped inverse square-root adjustment to effective class mass limited class imbalance. Final row weights ranged from 0.222 to 3.285 with mean one.

Open Pool marks 20,227 rows for proxy-generation timing. The full private experiment retained and downweighted them. The comparison subset with 173,501 rows also required checkable question provenance and screened against 150 reconstructed official topics or used previously verified independent prompts. Official questions and model labels from 30 reserved topics were excluded from official fitting and scoring. Both catalog curators used official statistics, and provenance is incomplete for some external questions. The two results are exploratory. Increasing the pool also increases model and source coverage and the number of training updates.

The SVM uses separate fixed hash spaces for character, word, word-shape and function-word/punctuation n-grams. Each view uses sublinear term frequency and row normalization. A linear hinge-loss classifier made one streamed external pass and then three official passes with identical official update order across arms. Feature hashing and incremental optimization differ from the earlier batch TF-IDF LinearSVC. The matched implementations in the tables allow the data comparison. The final full-catalog, screened and official-only arms shared feature extraction; together they took **7 minutes 20 seconds** on one CPU host and peaked at approximately **771 MiB** of process memory. Fivefold score matrices, label ordering, folds, row identifiers, both frozen baselines and physical source-row preservation were independently checked.
