# E39 unscored Kaggle T4 deployment probe draft

This probe tests deployment feasibility only. It uses the existing E39 fold-1
adapter and the fixed E24 + 1×E39 score rule. It does not fit a full adapter,
read validation labels, use the 30 reserved prompts for selection, or submit a
competition entry. The reviewed [deployment plan](../docs/e39_kaggle_t4_deployment_plan.md)
sets the purpose and hardware gate.

## Inputs and fixed checks

- `e39_probe_asset_manifest.json` lists the exact 12-file Qwen3-8B-Base
  revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` and nine fold-1
  adapter files by SHA-256. The base weights are Apache-2.0. Attach the pinned
  base snapshot and existing fold-1 adapter as **private** Kaggle inputs.
- Reuse the four E24 inputs already verified by its Notebook: the original
  E0c/RoBERTa package, E16 three-seed ModernBERT package, E24 E0c package,
  and offline CPU wheelhouse. The existing wheelhouse contains CPU PyTorch;
  it cannot supply E39 GPU inference.
- A private `e39_fold1_smoke_reference.json` must be generated once on the
  training server with `probe_e39_gpu.py reference`, using the already trained
  fold-1 adapter and the fixed E0 OOF split. The script selects three
  **fold-1 training-side** rows at zero-based official training ordinals
  `1, 3, 4`, hashes their text, and records 18 logits.
  Only row indices, hashes and logits enter this private reference; no text is
  copied into Git. Keep the reference in the private adapter input.
- The Kaggle GPU interpreter must provide CUDA PyTorch 2.6.0, Transformers
  4.51.3, PEFT 0.15.2, Accelerate 1.6.0 and bitsandbytes 0.45.5. The Notebook
  fails before model loading if these exact versions or T4×2 are unavailable.
  Prepare a separately hashed, offline **GPU** wheelhouse if the Kaggle image
  differs. Do not install from the network.
  The GPU environment probe also checks `pyarrow` 15–29 by writing and
  reading a one-row Parquet in memory before the long inference run.

The private smoke reference can be generated with the following command after
setting `CUDA_VISIBLE_DEVICES` to one free server RTX 3090 and using the
locked E39 runtime. The paths are relative to the project checkout:

```bash
python kaggle/probe_e39_gpu.py reference \
  --manifest kaggle/e39_probe_asset_manifest.json \
  --model-dir data/models/qwen3-8b-base-49e3418 \
  --adapter-dir outputs/e39_qwen3_base_fivefold_recovery1/fold_1/adapter \
  --train data/train.csv \
  --e0-oof outputs/e0_tfidf_svm/oof_predictions.csv \
  --output outputs/e39_probe_private/e39_fold1_smoke_reference.json
```

The output is Git ignored. Before running this command, verify that the chosen
GPU is free; the command loads a quantized 8B model. An adapter from a
different fold or full fit is a different experiment and fails the manifest
check.

## Unscored Notebook

`make_e39_t4_probe_notebook.py` generates
`probe_name_that_llm_e39_t4_unscored.ipynb` from the verified E24 setup. The
Notebook first validates the E24 packages and CPU interpreter, then checks
the GPU interpreter and hashes all E39 assets. It runs a three-text T4 smoke
test with **fixed** raw-logit absolute tolerance `0.05` and identical argmax.
For the initial attempt `DEVICE_MODE='single'`. If a single T4 cannot load,
the separate `DEVICE_MODE='dual'` attempt uses a fixed 18/18 Qwen layer split;
the entire smoke and 4,000-row test must be repeated. This split is an
untested engineering fallback until an actual T4 smoke passes. No score
determines the hardware choice.

The full proxy reads the official `train.csv` with `usecols=['text']`, repeats
those texts in a fixed order to 4,000 rows and gives them synthetic unique
IDs. The CPU process exports the unchanged three-stream E24 18-class score
matrix. The GPU process exports E39's 18 logits. The Notebook computes
`E24_scores + row_z(E39_logits)` with coefficient exactly 1, checks all values,
and writes `/kaggle/working/e39_unscored_probe/submission.parquet`. It reads
the Parquet back to verify 4,000 rows, `id,model` columns, ID order and labels.
The fixed feasibility limits are three hours for E39's 4,000-row inference,
at least 256 MiB actual GPU headroom at every sampled point, and eight hours for the **whole**
Notebook, leaving one hour below the competition's nine-hour cap.
Headroom uses the minimum driver-visible free bytes sampled after model load
and at each inference row, alongside PyTorch peak reservation; it includes
CUDA context and non-PyTorch allocations.

Run the private Notebook with T4×2 and Internet off using **Save Version →
Save & Run All**, then inspect logs and output. **Do not click Submit:** this
version uses synthetic IDs and is only an unscored load/time/schema test.
Passing it permits consideration of a separately audited full-data E39
adapter and a later, distinct submission Notebook; it does not itself make a
Kaggle prediction or prove a leaderboard gain.

No model weights, official texts, smoke reference, row-level scores, private
Dataset paths, or Kaggle credentials belong in Git.
