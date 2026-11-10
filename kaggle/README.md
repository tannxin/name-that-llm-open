# Kaggle submission notebook

The [competition](https://www.kaggle.com/competitions/name-that-llm/overview/abstract) requires an offline Notebook to write `/kaggle/working/submission.parquet` within nine hours. Its official input is `/kaggle/input/competitions/name-that-llm/test.csv`. The interactive test is only an example; the hidden rerun replaces it with approximately 4,000 rows.

## Prepare the model input

Run this from the project root **where the full model package was trained**:

```bash
python kaggle/prepare_artifact.py \
  --full-package outputs/final/full_package \
  --output outputs/kaggle_artifact
```

The command verifies the fitted source and model hashes, then copies the five exact Python source files, `requirements-final.txt`, and the full package. It excludes competition data and other experiment outputs. Upload the resulting directory as a private Kaggle Dataset and attach it to the Notebook. Do not substitute a GitHub checkout for the fitted source files: the inference loader checks their byte hashes.

If a single upload file is more convenient, archive the prepared directory:

```bash
python kaggle/archive_artifact.py \
  --artifact-dir outputs/kaggle_artifact \
  --output outputs/kaggle_artifact.zip
```

The ZIP contains `artifact_manifest.json`, `source/`, and `full_package/` at its root. The Notebook also accepts an attached Dataset containing this ZIP; it extracts into `/kaggle/working` and rechecks every file hash before inference. Keep sufficient working-space for both the ZIP and extracted model.

Import [submit_name_that_llm.ipynb](submit_name_that_llm.ipynb) into Kaggle, attach the private Dataset containing `kaggle_artifact.zip` and `kaggle_wheelhouse_cp313_linux_x86_64.zip`, select **CPU / accelerator None**, and disable Internet. Kaggle may mount those ZIPs as extracted `kaggle_artifact/` and `kaggle_wheelhouse_cp313_linux_x86_64/` directories. The Notebook detects either layout, verifies the model and all 35 wheel hashes, and invokes `final_pipeline.py submit`. It intentionally omits `--sample-submission`: the interactive three-row template may not match the hidden rerun. The final pipeline validates the full test IDs and labels and round-trips the Parquet file.

If Kaggle's installed scikit-learn, PyTorch, or Transformers version differs from the fitted package, the Notebook first looks for the mounted wheelhouse directory via `requirements-kaggle-offline.txt`; if needed, it falls back to the ZIP. It verifies the 35-wheel manifest and installs all 35 wheel files with `pip --no-index --no-deps` into a clean virtual environment. The wheelhouse omits `hf-xet`, which Hugging Face uses for Hub transfers; this inference loads local files only and sets `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_XET=1`. The Notebook probes local config and tokenizer loading before prediction. Kaggle's Python may fail to create a default virtual environment at `ensurepip`, so the Notebook creates one with `--without-pip` and uses the system pip's documented `--python` option to install into it. `WHEELHOUSE_DIR` and `WHEELHOUSE_ZIP` are optional overrides when autodetection is ambiguous. The wheelhouse contains **PyTorch 2.6.0+cpu** wheels for CPython 3.13 on Linux x86_64, so a GPU accelerator gives no benefit on that path. A server CPU benchmark processed 256 passages in 73.99 seconds, suggesting roughly 20 minutes for 4,000 passages on comparable hardware. Kaggle's hidden rerun of the original Notebook completed successfully.

## Fixed normalized-score candidate

The second-round [score comparison](../docs/e10_fusion_normalization.md) keeps the same fitted E0c and RoBERTa models and changes only the final combination: each model's 18 class scores are centered and scaled within the test row, then summed equally. The standalone implementation is [submit_normalized.py](submit_normalized.py); it calls the existing package loader, so the five training-time source hashes are still checked. `make_normalized_notebook.py` generates [submit_name_that_llm_zscore.ipynb](submit_name_that_llm_zscore.ipynb) from the original verified Notebook and embeds the exact normalized submission source with its SHA-256. Attach the same private model Dataset and CPU wheelhouse, with Internet off and accelerator None. The original rank-based Notebook stays available as the Kaggle control.

The [normalized Notebook Version 3](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365857241) completed Kaggle's hidden rerun with **0.401 public balanced accuracy**. The original [rank-based Version 2](https://www.kaggle.com/code/tanxin357/name-that-llm-e0c-roberta-blend?scriptVersionId=365821422) scored **0.376**. These are public hidden-test scores (+0.025 for Version 3), separate from the 0.3995 and 0.3833 development OOF scores. The private leaderboard result is not yet known.
