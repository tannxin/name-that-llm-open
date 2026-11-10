# E2 raw MPNet supervised comparison

`microsoft/mpnet-base` was fine-tuned with `run_finetune_encoder.py` on the same 2,160 development answers and five E0 prompt-grouped folds. The 30 prompt holdout groups were not tokenized, trained on, or scored. The pretrained checkpoint was pinned to [`6996ce1e91bd2a9c7d7f61daec37463394f73f09`](https://huggingface.co/microsoft/mpnet-base/tree/6996ce1e91bd2a9c7d7f61daec37463394f73f09). Its `model.safetensors` was obtained via the ModelScope mirror and verified against the [official SHA-256](https://huggingface.co/microsoft/mpnet-base/blob/6996ce1e91bd2a9c7d7f61daec37463394f73f09/model.safetensors): `f8eb85dab52341af018fb8cd325a9d83fdb00871c1b2cb1d445d42b289c1d45e`. The official repository currently has no license field in its model API metadata; no pretrained or fine-tuned weights are included in Git.

The fixed schedule used 12 epochs per fold, a 256-token cap, masked mean pooling without L2 normalization, a dropout and linear 18-class head, encoder learning rate `2e-5`, head learning rate `1e-3`, batch size 16, seed 42, and inverse fold-training class-frequency weights. The final epoch was scored once per fold; validation results did not select an epoch. Fold checkpoints were not retained. Full parameters and software versions are recorded in `outputs/e2_finetune/mpnet_base_meanpool_12ep/metrics.json` and `run_config.json`.

| Fold | Supervised raw MPNet | Frozen raw MPNet |
| --- | ---: | ---: |
| 1 | 0.261574 | 0.303241 |
| 2 | 0.293981 | 0.296296 |
| 3 | 0.337963 | 0.266204 |
| 4 | 0.287037 | 0.298611 |
| 5 | 0.293981 | 0.250000 |
| **Pooled OOF** | **0.294907** | **0.282870** |

The E0 TF-IDF baseline scored 0.281019 and the E0c TF-IDF plus style baseline scored 0.342593 on the same development folds. This supervised MPNet run is 0.047685 below E0c. Its 0.012037 gain over frozen MPNet is small and needs paired prompt-level uncertainty analysis before being treated as reliable. The OOF file has all 2,160 E0 development IDs with matching fold assignments and no holdout prompt IDs; recomputing balanced accuracy from its predictions gives 0.294907.

The loader reports newly initialized MPNet pooler weights. The classifier consumes `last_hidden_state` with attention-masked mean pooling, so the pooler output is unused. No scheduler or skipped-step warning appeared in this run's log.
