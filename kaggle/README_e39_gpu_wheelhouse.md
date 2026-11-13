# E39 GPU wheelhouse preparation (unscored)

This is a separate CPython 3.13 Linux x86-64 CUDA 11.8 environment for the
unchanged E39 fold-1 deployment probe. Kaggle's observed GPU image has
Python 3.13.16, Torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2,
Accelerate 1.15.0 and no bitsandbytes. Those versions do not satisfy the
locked E39 inference code. The existing E24 CPU environment stays separate.

## Local preparation

The Git-safe [spec](e39_gpu_wheelhouse_spec.json) pins the E39 direct packages,
Python ABI and Linux tags. The existing 35-wheel CPU ZIP is verified by its
SHA-256; its `torch-2.6.0+cpu` wheel is excluded. Compatible wheels are reused
at their original hashes. The script resolves remaining wheels through the
pinned Aliyun PyPI transfer mirror. The CUDA Torch wheel is separately checked
against the published SHA-256 from the official PyTorch CUDA 11.8 index; the
bitsandbytes Linux wheel is checked against its official PyPI SHA-256. Every
newly transferred non-Torch wheel is independently checked against official
PyPI release JSON SHA-256 and byte length. Reused wheels are checked against
the original CPU ZIP manifest before and after pip. The builder performs a
pip offline dry run with **only** the new wheelhouse,
then writes a full transitive `--require-hashes` lock and a per-file SHA-256
manifest. All files are under Git-ignored `outputs/`.

The preparation host is Windows, while Kaggle is Linux. Pip's cross-target
wheel flags do not change how it evaluates Linux-only dependency markers.
The spec therefore pins all 11 CUDA wheels and Triton required by the exact
Torch wheel as explicit requirements. The builder compares those pins to
Torch METADATA and independently verifies every selected wheel's dependency
closure against a Linux CPython 3.13 marker environment. A Windows-only pip
dry run alone is not accepted as proof of closure.
The same marker check explicitly pins `hf-xet==1.1.10`, required by the locked
Hugging Face Hub wheel on Linux.

```text
python kaggle/prepare_e39_gpu_wheelhouse.py download
python kaggle/prepare_e39_gpu_wheelhouse.py verify
python kaggle/prepare_e39_gpu_wheelhouse.py refresh-primary-index
python kaggle/make_e39_gpu_env_test_notebook.py
```

The primary Torch wheel must be exactly
`torch-2.6.0+cu118-cp313-cp313-linux_x86_64.whl`. If the official simple index
provides a SHA-256 fragment, the script compares it to the downloaded file and
records `primary_torch_index_hash_verified=true`. The exact official Torch
SHA-256 `8d30eb2870ffe05d81ec513bdb08c0f2bab9fd1bd4fbc6e5681fad855c7b99e3`
and its 848,630,288-byte length are also pinned in the spec.
If the official index is temporarily unreachable during the main build, the
`refresh-primary-index` action first verifies the completed wheelhouse, then
requires a live HTTPS index SHA match and atomically updates its manifest;
rerun the Notebook builder afterward to embed the new manifest SHA.
The public mirror
transfer helper `fetch_e39_torch_wheel_resumable.py` uses verified HTTP byte
ranges and only publishes a wheel after the full official size and SHA match.
It must run as the sole wheel writer; a failed transfer retains parts for
inspection. The bitsandbytes Linux wheel's published SHA-256 is pinned too.
A failed pip resolution, missing wheel, hash
change, duplicate distribution or CPU Torch fails closed.

The completed local wheelhouse has 52 wheels totaling 3,157,594,345 bytes
(2.94 GiB); 47 distributions are selected by the Linux cp313 lock. Its
manifest SHA-256 is
`a9a373ba8cf9d688b7730bdef811d4bd578416bda66bc1200f35af6accc881b8`,
including a successful live official PyTorch index check. All large wheels
remain in Git-ignored `outputs/`; only this spec, scripts, README and the
unscored Notebook are suitable for Git review.

## Private Dataset archive

`python kaggle/package_e39_gpu_wheelhouse.py package` creates the Git-ignored
`outputs/e39_gpu_wheelhouse_cp313_cu118.zip`. The packager pins the final
manifest byte hash, re-verifies the wheelhouse, and includes exactly 54 flat
members: its 52 wheels, `e39_gpu_wheel_manifest.json`, and
`requirements-e39-gpu-locked.txt`. Every member is `ZIP_STORED` with fixed ZIP
metadata and is checked against its expected size and SHA-256 before the
archive is published. `python kaggle/package_e39_gpu_wheelhouse.py verify`
performs the same checks again without changing the archive.

The verified archive is 3,157,688,910 bytes with SHA-256
`905bcd9be4a2d4d231fd3ff3eb631ada31856009f27c08467d7528208f559f91`.
It contains no model, adapter, competition data, logs, cache or download
segments. Kaggle's Dataset documentation says an uploaded ZIP is unpacked
server-side. After a private upload, inspect the mounted Dataset to confirm
the 54 flat files are present before running the unscored Notebook.

Cross-platform pip resolution, Linux dependency markers, wheel tags and the
finished manifest have passed an independent local static review. No Kaggle
asset is created by these local commands; Kaggle runtime compatibility remains
untested.

## Small offline test

[The generated Notebook](probe_e39_gpu_env_offline_unscored.ipynb) embeds the
completed local manifest SHA-256. Before a wheelhouse download completes, it
contains a placeholder and refuses to run. Attach **only** the private wheel
Dataset to a fresh T4 x2, Internet-off Notebook. The Notebook hashes every
mounted wheel, checks at least 10 GiB free in `/kaggle/working`, creates a
fresh virtual environment without system site packages, installs with
`--no-index --no-cache-dir --require-hashes`, and runs `pip check`. It imports
the exact E39 stack and `Qwen3ForSequenceClassification`, asserts CUDA 11.8,
then performs a tiny NF4 forward on each T4. It reads no competition files,
model weights, adapter, validation labels or public leaderboard; it writes no
submission and must never be submitted. A pass only proves package and tiny
kernel compatibility. The full fold-adapter smoke and 4,000-row proxy remain
separate deployment gates.

The private wheel Dataset is read-only. Keep its wheels there rather than
copying them into Kaggle's approximately 19.5 GiB `/kaggle/working`. The
installed environment and pip temporary files use that working space. Measure
actual installation peak and remaining free space; the direct-mounted Qwen
Base must also stay in `/kaggle/input` for later testing. Do not replace the
locked runtime with Kaggle's newer packages or add the CPU Torch wheel to the
GPU wheelhouse.

## Static review before any Kaggle action

1. Verify the source spec, script and Notebook contain no model/data/score
   access or submission action; check Python syntax and Notebook outputs empty.
2. Verify source CPU ZIP hash and every reused wheel against its original
   manifest, with CPU Torch excluded. Verify exact Torch CUDA and bitsandbytes
   published hashes; inspect GPU Torch wheel metadata for CUDA/triton
   dependencies.
3. Verify the downloaded full dependency closure against Linux CPython 3.13
   markers as well as offline pip resolution,
   unique distribution versions, cp313 Linux tags and exact hashes; inspect
   `primary_torch_index_hash_verified` and any unresolved origin URL.
4. Review the Kaggle Notebook for isolated venv, no network, exact version/CUDA
   assertions, both T4 checks, `pip check`, tiny NF4 computation and working
   storage gate. A passing tiny test does not authorize a scored submission.

