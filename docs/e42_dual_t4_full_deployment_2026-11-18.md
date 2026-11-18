# E24 + E42 completed 4,000-row inference on two Kaggle T4 GPUs

The fixed E24 ensemble and the Qwen3-14B-Base fold-1 adapter completed offline inference in **3 hours 3 minutes**, within the eight-hour deployment budget. An independent audit reproduced **all 4,000 output labels** from the two saved score matrices. This clears the deployment check for planning a separately verified full-data training run. [Deployment protocol](e42_t4_deployment_probe_plan.md)

The fixture cycled through 2,160 unlabeled development rows covering 120 prompts and assigned 4,000 unique synthetic IDs. Inference used only this fixture and excluded all 30 held-out prompts. The fixed E24 artifacts had previously been trained using the full official training set. The deployment check used the same pinned model and adapter as the three-text comparison. [Dual-T4 smoke check](e42_dual_t4_smoke_2026-11-18.md)

| Measurement | Result |
|---|---:|
| E24 CPU phase | 6,894.72 s |
| E42 dual-T4 phase | 3,102.02 s |
| End-to-end probe time | 10,959.97 s |
| Kaggle job duration | 10,983 s |
| Peak reserved VRAM, T4 0 / T4 1 | 7.07 / 6.26 GiB |
| Minimum sampled free working disk | 13.28 GiB |

Both matrices contained 4,000 rows in the expected ID order and finite scores for all 18 classes. The audit standardized each E42 row across its 18 logits, added it to the fixed E24 scores at unit weight and selected the highest-scoring class. The resulting ordered IDs and labels matched the saved Parquet file exactly. Its SHA-256 was `340c7c8da671684c6f2a81fe59612e533b2d98a8e6318eb8ed07e5250e3a8482`.

All 18 classes occurred among the predictions. Adding E42 changed the top-ranked class on 0 of the 4,000 proxy rows. E24 had trained on these development prompts, and the replay used no labels, so this agreement measures deployment behavior only. The separately audited five-fold balanced accuracy for E24 + E42 was 0.443519. The replay generated no competition submission. The project's best public leaderboard submission remains E24 + E39 at 0.445. [Five-fold audit](e42_qwen3_14b_base_recovery2_fivefold_results.md)
