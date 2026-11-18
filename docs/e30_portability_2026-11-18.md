# E30 cross-device portability check (2026-11-18)

E30-A did not pass its preregistered offline promotion gate. A Kaggle T4×2, Internet-off, three-text fold-1 smoke check loaded the pinned model and adapter, but its raw logits differed from the archived 3090 reference by a maximum absolute value of **0.011469**, above the fixed **0.002** limit. No full-data fit or diagnostic Kaggle submission followed.

A reload on the original 3090 reproduced all three archived logits exactly (maximum difference **0.0**). A separately reviewed, one-shot 3090 capture then produced 12 fixed fold-1 *training* text references: a finite `12×4096` pooled feature array and `12×18` logit array. Its private and shared-storage bytes matched, GPU0 was released, and the guarded Pod-local storage change was below 1 MiB. This capture computed no validation score, used no test labels, and made no submission.

The separately reviewed Kaggle T4 check of the 12 fixed training inputs completed in private Notebook Version 4 (`370925541`). The pinned Base and adapter, tokenization, pooling positions, and classifier head were verified. The maximum absolute differences from the 3090 reference were **0.011442** for the `12×18` raw logits and **0.078125** for the `12×4096` pooled features. Both exceeded the unchanged `rtol=0, atol=0.002` gate. The run stopped before comparing the three development OOF texts. It read no competition test data, computed no accuracy, and made no submission.

The difference is in the encoder output. Device-dependent attention or NF4 matrix kernels are hypotheses, not established causes. A further diagnostic would compare fixed training texts with the same explicitly selected attention implementation on both devices, using a new reviewed no-score protocol. E30-A's offline No-go status and the original numerical gate remain unchanged.

## Did the measured differences change the first-choice class?

For the 12 fixed training texts in the two-device comparison, **all 12 E30 first-choice classes were preserved**. This follows from the archived 3090 logits and the global maximum absolute error of 0.01144218445 reported by the T4 run. If each logit can move by at most *e*, the gap between the largest and second-largest logits can shrink by at most *2e*. The smallest 3090 gap among the 12 texts was **0.209971**, comfortably above the conservative bound of **0.022885**. The calculation rounded the error bound upward before applying the strict comparison.

This check used the T4 run's saved error summary and compared class decisions without reading any target labels. It covers the E30 component on these 12 training texts. Accuracy, decisions after fusion with E24, and behavior on held-out or hidden texts remain unmeasured here. The raw-logit difference still exceeds the preregistered deployment tolerance of 0.002.
