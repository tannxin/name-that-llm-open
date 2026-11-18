# Replaying the public 0.684 ensemble on 64 unlabeled texts

On November 18, 2026, private Kaggle Version 2 selected the original ensemble path (`FAST_COMMIT=False`) and stopped after **27.2 seconds**. Both T4 GPUs had **14,912 MiB free each** at startup. The first input-discovery cell could not find `ntl_open_pool.parquet`. The failure occurred before any classifier ran. This attempt produced no predictions, submission, or leaderboard score. The published **0.684** belongs to [the author’s original V1](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline?scriptVersionId=368242626).

## What changed between our private runs

The successful three-text preview (private Version 1) listed **Name That LLM Open Pool: 224k Clean Passages** and displayed `ntl_open_pool.parquet` at **112.85 MB**. The catalogue entry for Open Pool v17 (`datasetVersionId=22123604`) matches the visible filenames and size, **112,851,698 bytes**. This identifies a match for our preview. The author’s historical attachment bytes remain unverified. [Three-text preview results](public_0684_v1_replay_2026-11-18.md)

Private Version 2 omitted the entire Open Pool attachment. Comparing the two input lists also showed that Version 2 omitted **Name That LLM Artifacts** and both **NTL Ours Qwen3-14B LoRA** attachments (including v4), and added **NTL ATLAS Qwen3 8B v1**. The Open Pool omission explains the first assertion. The run stopped too early to test the other attachments or measure ensemble accuracy and runtime. The author’s `atlas_mode=off` setting was preserved.

## CPU verification of the restored inputs

Kaggle CPU catalogue version `370983986` ran with internet disabled and finished in **152.4 seconds**. It reported `INPUTS_PRESENT` for **34 of 34** expected sources, with zero errors in the mount and startup-input checks. The corrected 64-text diagnostic removes stale attachment metadata so it can inherit these inputs. All 31 notebook cells match the failed diagnostic exactly.

The next execution step is to run the corrected diagnostic on two T4s and verify each active classifier from its logs and outputs. A second GPU run and a byte-for-byte match to the author’s historical assets remain unverified.
