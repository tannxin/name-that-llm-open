# Qwen3-8B with Prvsiyan v9: preregistered resource check

Status: **resource gate passed; fivefold performance has not run** (2026-11-17 UTC). This experiment tests a fixed external-data intervention on the existing E39 Qwen3-8B-Base model. The best submitted Kaggle score remains E24+E39, public balanced accuracy **0.445**; this resource check produced no new leaderboard score.

## Locked protocol

- Model: `Qwen/Qwen3-8B-Base` revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, the existing E39 tokenizer, NF4 LoRA configuration, and unchanged trainable 18-class score head. Every later fold starts from the pinned Base, not an E39 adapter.
- Data: each official outer training fold has 1,728 responses from 96 development prompts. Append the same 3,620 accepted `split=new` responses from [Prvsiyan Extended Name That LLM Dataset v9](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/version/9), for 5,348 training rows per fold. These external rows cover 17 of 18 classes; no accepted Qwen 3.8 Max row is available. The 24 validation prompts per fold, 30 previously used holdout prompts, and hidden test are excluded from fitting.
- Weights: each covered class receives fixed external total weight 24 beside official total 96; official Qwen responses receive weight 1.25. Microbatch 1, gradient accumulation 8, three epochs, seed 42, and the archived E39 optimizer/schedule remain fixed.
- Assessment after **all five** 432-row OOF folds: compare the fixed E24 + 1×mixed-Qwen fusion against fixed E24 + 1×official-only E39. Promotion requires balanced accuracy at least +0.01, at least four of five folds nondecreasing, and every class recall drop strictly below 0.05. No single-fold or public leaderboard tuning is allowed.

The source generator used statistics of the full official training-response distribution to choose length/style constraints. Thus future OOF results are **exploratory**, even though the accepted prompt bodies differ from the official development prompts. The source subset and weights were fixed before this resource test.

## Independent resource audit

One unscored fold-1 training-side probe used 5,348 rows for one epoch, then 4,000 repeated **unlabelled training-side** texts for inference. The independent audit verified the pinned protocol, model file hash inventory, package RECORD hashes, Pod/GPU identity, no validation or holdout evaluation, and the report/control SHA chain. It recomputed the thresholds from the raw report numbers.

| Measure | Observed | Preregistered gate |
| --- | ---: | ---: |
| First epoch | 59.17 minutes | ≤90 minutes |
| Three-epoch projection | 2.96 hours | ≤4 hours |
| Successful optimizer updates | 663/669 (99.1%) | ≥95% |
| Training reserved GPU memory | 8.00 GiB | <22 GiB |
| 4,000-row inference | 10.35 minutes | ≤3 hours |
| Inference reserved GPU memory | 14.80 GiB | <22 GiB |

The one reported training loss was finite. The report records `validation_rows=0`, `validation_scores_computed=false`, and `holdout_evaluated=false`. Resource report SHA-256: `420ff59a9cad11704f872edef20a39094ccb9d745232e9f8ada85d8d99955761`; locked protocol SHA-256: `59cc0da4815c7b8b04365683ec5f32102434946eb8a94268447872cb61fa868b`. [Machine-readable aggregate](q8_v9_mixed_v1_resource_aggregate.json) contains the independent recalculation.

This is **resource-only GO**. It does not authorize full-data training or Kaggle submission. The fivefold runner and execution guard require separate review; all five OOF folds and a separate Kaggle T4×2, Internet-off, nine-hour deployment check would be required for a candidate.

The Kubernetes service account cannot read the current node eviction threshold or authoritative Pod ephemeral usage. This run used a reviewed local-storage watchdog and one PVC copy of model/cache/output, but no periodic watchdog telemetry artifact exists. Successful completion and sampled free-space checks do not establish the node's eviction margin. Future runs must retain the conservative storage guard and preserve failure evidence.
