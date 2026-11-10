# E24 audited external E0c replacement

Run the fixed full fit only after the E24 source audit and fivefold outputs are present:

```bash
python fit_e24_full_e0c.py
python kaggle/prepare_e24_artifact.py
python kaggle/make_e24_notebook.py
```

Upload only `outputs/e24_kaggle_artifact_v1.zip` as a new private Kaggle Dataset. Import `kaggle/submit_name_that_llm_e24.ipynb` as a separate Notebook. Attach that Dataset, the existing E0c/RoBERTa package, the E16 ModernBERT package, and the offline wheelhouse. Keep Internet off and accelerator None. The Notebook verifies mounted files, loads all models, checks E24's smoke scores and official 18-class order, and writes the competition Parquet.

The fixed fivefold score was 0.425000 versus 0.409259 for E16; this is exploratory and not a Kaggle score. The external source is [Prvsiyan Extended Name That LLM Dataset v9](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/version/9), CC BY 4.0. Its generation protocol used global official train length statistics. The raw answer text and original API envelopes stay out of the model ZIP and repository. See [E24 handoff](../docs/e24_kaggle_handoff.md) for hashes and validation.
