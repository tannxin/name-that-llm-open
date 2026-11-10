# E16 three-seed ModernBERT Kaggle candidate

This is a separate offline submission candidate. It retains the existing
full-data E0c and RoBERTa package and adds three full-data ModernBERT-base
models trained with seeds 42, 17, and 73. For each test row, it standardizes
each model's 18 scores across classes, averages the three ModernBERT scores
into one stream, and adds that stream to the E0c and RoBERTa streams with equal
weight. The fixed development OOF balanced accuracy was 0.409259, compared
with 0.399537 for the E0c + RoBERTa candidate. This is not a Kaggle score.

## Package the ModernBERT models

Run from the project root on the machine holding all three verified full-data
packages, after each has a successful `smoke_verification.json`:

```bash
python kaggle/prepare_e16_artifact.py \
  --seed42 outputs/e16_modernbert_full_seed42_v2 \
  --seed17 outputs/e16_modernbert_full_seed17_v2 \
  --seed73 outputs/e16_modernbert_full_seed73_v2 \
  --output outputs/e16_kaggle_artifact_v2 \
  --zip outputs/e16_kaggle_artifact_v2.zip
```

The packager checks the fixed label order, revision, training-data hash,
source hash, runtime, training settings, artifact hashes, and reload smoke
result. The ZIP stores each model file without recompression and is checked
again against the manifest. It excludes training data and unrelated outputs.

## Run the separate Notebook

Import [submit_name_that_llm_e16.ipynb](submit_name_that_llm_e16.ipynb) into
Kaggle and attach three private inputs:

1. The existing `kaggle_artifact.zip` with E0c and RoBERTa.
2. The existing `kaggle_wheelhouse_cp313_linux_x86_64.zip`.
3. The new `e16_kaggle_artifact_v2.zip`.

Set **Internet off** and **accelerator None**. The Notebook accepts Kaggle's
extracted Dataset layout or a mounted ZIP. Its config cell has optional paths
if autodetection finds multiple candidates. It verifies every mounted model
file and source hash, probes local tokenizer/config loading with the selected
Python runtime, and invokes the standalone
[submit_e16_three_seed.py](submit_e16_three_seed.py). That script checks each
ModernBERT package again, runs a three-text reload smoke comparison, validates
the official test IDs and label order, then writes and reopens
`/kaggle/working/submission.parquet`.

The Notebook uses the existing 35-wheel CPU wheelhouse when Kaggle's runtime
does not match the trained package. It inherits the proven v3 dependency setup
but does not change the v3 Notebook or its model Dataset. A full local offline
CPU run of 4,000 fixture rows completed in 83 minutes 24.582 seconds; the
output's columns, IDs, and label values passed independent checks. Kaggle's
runtime may differ. The exact files, hashes, validation record, and manual
submission steps are in [the E16 handoff](../docs/e16_kaggle_handoff.md).
