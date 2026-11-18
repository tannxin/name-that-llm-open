# Replaying the public 0.684 ensemble on 64 unlabeled texts

On November 18, 2026, private Kaggle Version 2 selected the original ensemble path (`FAST_COMMIT=False`) and stopped after **27.2 seconds**. Both T4 GPUs had **14,912 MiB free each** at startup. The first input-discovery cell could not find `ntl_open_pool.parquet`. The failure occurred before any classifier ran. This attempt produced no predictions, submission, or leaderboard score. The published **0.684** belongs to [the author’s original V1](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline?scriptVersionId=368242626).

## What changed between our private runs

The successful three-text preview (private Version 1) listed **Name That LLM Open Pool: 224k Clean Passages** and displayed `ntl_open_pool.parquet` at **112.85 MB**. The catalogue entry for Open Pool v17 (`datasetVersionId=22123604`) matches the visible filenames and size, **112,851,698 bytes**. This identifies a match for our preview. The author’s historical attachment bytes remain unverified. [Three-text preview results](public_0684_v1_replay_2026-11-18.md)

Private Version 2 omitted the entire Open Pool attachment. Comparing the two input lists also showed that Version 2 omitted **Name That LLM Artifacts** and both **NTL Ours Qwen3-14B LoRA** attachments (including v4), and added **NTL ATLAS Qwen3 8B v1**. The Open Pool omission explains the first assertion. The run stopped too early to test the other attachments or measure ensemble accuracy and runtime. The author’s `atlas_mode=off` setting was preserved.

## CPU verification of the restored inputs

Kaggle CPU catalogue version `370983986` ran with internet disabled and finished in **152.4 seconds**. It reported `INPUTS_PRESENT` for **34 of 34** expected sources, with zero errors in the mount and startup-input checks. The corrected 64-text diagnostic removes stale attachment metadata so it can inherit these inputs. All 31 notebook cells match the failed diagnostic exactly.

The table below records how far the corrected GPU run progressed.

## Local replay of the three fitted CPU classifiers

The original `core_votes` function ran on the same 64 unlabeled texts in an offline WSL2 process. With version 22 of the prefit archive, the call took **7.09 seconds**. Each classifier produced 64 finite 18-class vectors. The three label orders matched and every top choice was unique.

| Classifiers | Same first choice | Different first choice |
| --- | ---: | ---: |
| `svc4` and `tfidf` | 31 | 33 |
| `svc4` and `distort` | 22 | 42 |
| `tfidf` and `distort` | 17 | 47 |

All three chose the same class for 11 texts. Exactly two agreed for 37 texts, and all three chose different classes for 16 texts. These counts establish that the classifiers make different decisions on this sample. Measuring their contributions to accuracy requires labeled examples held out from each classifier's training. The local timing covers only these three classifiers. A later dual-T4 run tested the neural path and reached the Qwen3-8B decoder.

## Result of the corrected dual-T4 run

The corrected private version (371005916) ran for **2,028.2 seconds**. It read the restored Open Pool and reached `comp_stack.py` with 64 unlabeled inputs. Two contributions to the partial ensemble are confirmed by the accumulator log.

| Component | Evidence on this run |
| --- | --- |
| StyleGram | Added a vote on all 64 rows at weight 1.516. |
| Stylometry LightGBM | Added a vote on all 64 rows at weight 0.063. |
| GLM-9B | Computed features and grouped the 64 texts as singletons. The plain head had zero weight. The log showed a Gaussian fit but no vote from that head. |
| Qwen3-8B decoder | Failed when loading the checkpoint, before this branch could score the texts. |

The original checkpoint finder sorted every matching configuration path and selected the first one with the requested model type and hidden size. A LoRA attachment supplied a matching configuration directory without base weights. The official Qwen3-8B-Base input was mounted, but this decoder resolved to the configuration-only directory. The worker exited with code 1 before the remaining ensemble and final inference could run. The 64-row run produced no final predictions or leaderboard score.

The targeted repair is to bind the failing decoder to the specific 8B Base mount and verify its configuration, shard index, and every referenced shard before the expensive fit starts. Other adapters require independent evidence of the base they use. The logged voting weights describe execution only. Measuring contribution to accuracy requires labeled, held-out predictions.

## Replay with the verified Qwen3-8B-Base checkpoint

A separate offline Kaggle CPU run read all 16,381,516,776 bytes from the five official 8B Base weight shards. All five SHA-256 digests, the configuration and the shard index matched. The CPU run took 323 seconds.

The next private dual-T4 run (371045492) used the same 64 unlabeled texts and 34 inputs. The log recorded the intended 8B checkpoint path. Qwen3-8B feature extraction finished in 338 training batches and 8 diagnostic batches. Its plain classification head had zero weight in the author's configuration. The log recorded a Gaussian fit. StyleGram and stylometry LightGBM each added a vote for all 64 rows.

The component stack then requested a Qwen2 checkpoint with hidden size 3,584 for the Qwen2.5-7B branch. Its finder found no matching configuration under the attached inputs and the worker exited with code 1. Kaggle ended the run after 2,540.4 seconds. Final inference never started. The original author's public score remains separate from these unlabeled checks.

The next CPU check must inspect checkpoint discovery across all requested backbones together, including the adapter-to-base relationship and the weight files named in each index. Adding a replacement model requires evidence of its variant and revision.
