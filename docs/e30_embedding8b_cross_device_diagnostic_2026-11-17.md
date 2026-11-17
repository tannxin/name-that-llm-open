# E30 embedding 8B cross-device diagnostic — 2026-11-17

The fixed E24+E30-A fusion reached **0.448611** development out-of-fold (OOF) balanced accuracy, but failed the previously registered conservative comparison with E28 on one class-recall gate. The user authorized one *diagnostic* Kaggle submission, subject to a separate deployment-fidelity check. **No new E30 submission or public score exists.**

The private Kaggle T4×2, Internet-off, no-score smoke loaded the pinned fold-1 Base and adapter and evaluated three fixed development OOF texts. Its largest raw-logit difference from the locked 3090 reference was **0.011469364**, above the fixed **0.002** absolute tolerance. The Kaggle session was stopped without full-data fitting, competition-test prediction or submission.

A one-shot reload on the original 3090 then used the same fold-1 Base, adapter and three fixed texts. The audited report recorded **3×18** logits, maximum absolute difference **0.0**, identical argmax classes, **8,029,995,008 bytes** peak reserved GPU memory, and **528.59 seconds** total supervised run time. The worker and supervisor reports agree on the same run, and their copied bytes passed independent SHA-256 checks. They explicitly report no accuracy computation, validation-label use, competition-test read or submission. The report does not retain a separate raw output matrix, so its numeric difference cannot be independently recomputed from row-level logits.

This establishes that the locked 3090 reference can be reproduced on the source platform. It does **not** isolate why Kaggle T4 differs; GPU kernels, precision and runtime remain possible causes. The original **0.002** fidelity gate remains in force. Next is a separately specified, unscored T4 portability diagnostic. E30 stays blocked from full-data fitting and Kaggle submission until a fixed deployment passes its smoke and complete 4,000-row, nine-hour resource check.

Only aggregate diagnostics are published here. Model weights, training data, row-level OOF and raw logits stay outside Git.
