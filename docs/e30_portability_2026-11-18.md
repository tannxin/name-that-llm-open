# E30 cross-device portability check (2026-11-18)

E30-A did not pass its preregistered offline promotion gate. A Kaggle T4×2, Internet-off, three-text fold-1 smoke check loaded the pinned model and adapter, but its raw logits differed from the archived 3090 reference by a maximum absolute value of **0.011469**, above the fixed **0.002** limit. No full-data fit or diagnostic Kaggle submission followed.

A reload on the original 3090 reproduced all three archived logits exactly (maximum difference **0.0**). A separately reviewed, one-shot 3090 capture then produced 12 fixed fold-1 *training* text references: a finite `12×4096` pooled feature array and `12×18` logit array. Its private and shared-storage bytes matched, GPU0 was released, and the guarded Pod-local storage change was below 1 MiB. This capture computed no validation score, used no test labels, and made no submission.

The next check is a separately reviewed, no-score Kaggle T4 comparison of those 12 fixed inputs with `rtol=0, atol=0.002`. It runs the original three-text OOF smoke check only if all 12 training canaries pass. The 0.002 limit remains fixed; this diagnostic does not alter E30-A's offline No-go status.
