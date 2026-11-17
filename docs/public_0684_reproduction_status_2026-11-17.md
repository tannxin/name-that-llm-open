# Public 0.684 approach: reproduction status

Status as of 2026-11-17: **input and CPU environment preflight passed; the published 0.684 result has not been reproduced or independently verified.** This report concerns the third-party approach, not a new Name That LLM leaderboard submission.

## Completed check

A separate private Kaggle Notebook ran the version 3 gate with CPU only and Internet off. The original 55-cell reference Notebook was unchanged and was not executed. The gate returned `GO_CPU_PREFLIGHT_ONLY` with zero errors:

| Check | Result |
| --- | ---: |
| Mounted input roots | 42/42: 24 datasets, 8 Notebook outputs, 10 model instances |
| Declared dataset files checked by exact size | 1,648/1,648 |
| Files in three Qwen Base inventories checked by exact size | 47/47 |
| Fixed source markers | 49/49 |
| Offline wheel files checked by exact size | 6/6 |
| Scratch space | Passed: `/tmp` strictly above 29.522 GiB free; `/kaggle/temp` at least 8 GiB; `/kaggle/working` at least 2 GiB |

The gate installed the six declared wheels offline into a disposable directory, checked imports with CUDA hidden, and removed that directory. It inspected file paths, sizes, and disk space. It did not open model weights, read competition rows or labels, train, predict, score, or use a GPU. File size agreement does not prove that every attachment has the same bytes as the third-party run; full checksum coverage is unavailable.

## Diagnostic history

- Version 1 stopped before its preflight because Windows line-ending conversion changed the embedded manifest's SHA256. No reference cells ran.
- Version 2 verified all declared inputs but returned `NO_GO` because `/kaggle/temp` did not yet exist in a fresh session. The reference Notebook creates that directory during setup.
- Version 3 restored the script and manifest from byte-exact embedded payloads. It created only the empty `/kaggle/temp` directory when needed, then checked the actual free space. This private CPU preflight returned `GO_CPU_PREFLIGHT_ONLY`.

## Source identity, attachments, and validation boundary

The [original Kaggle page](https://www.kaggle.com/code/wadigongren/name-that-llm-0684-clean-baseline/comments) displays a **0.684 public score for V1**. The source available for this audit is a separate 55-cell **V3** mirror, `lbjyang/ntl-684-ozertuu-mirror`, preserved in a [fixed GitHub archive](https://github.com/SSanjay0614/Name-That-LLM-Kaggle/blob/55d80acdccc7a4e3d30b3acf9a0838a1a9604f4b/public_new3/ntl-684-ozertuu-mirror.ipynb) (SHA-256 `c1528ae5ca75d17479038cd1b9ebe8a1d7dcfcc2a3761b3eb11a33786343f5b7`). We could not obtain the original V1 source and input manifest for a cell-by-cell or attachment-version comparison. The V3 mirror and its CPU input preflight therefore do **not** establish source identity with the scored V1, reproduce 0.684, or prove a full hidden-test run within nine hours. The mirror's three-row public preview uses fixed predictions without loading the neural branches; its short visible runtime cannot stand in for full inference.

The archived [V3 input manifest](https://github.com/SSanjay0614/Name-That-LLM-Kaggle/blob/55d80acdccc7a4e3d30b3acf9a0838a1a9604f4b/public_new3/metadata.json) references 24 datasets, 8 Notebook output sources, and 10 model instances. The model references specify versions; the dataset and Notebook output references do not. Current mounted file sizes are only a snapshot, not proof of the scored V1 bytes or historical versions. The dataset license snapshot contains `Other` and `Unknown` entries; the largest [fitted-asset pack](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) is listed as `Other` and includes separately licensed components. A Notebook code license does not grant redistribution rights to its attached weights. Keep all third-party weights and data outside this project's public MIT repository.

The archived [V3 code](https://github.com/SSanjay0614/Name-That-LLM-Kaggle/blob/55d80acdccc7a4e3d30b3acf9a0838a1a9604f4b/public_new3/ntl-684-ozertuu-mirror.ipynb) references **all 2,699 official training rows across 150 prompts** for fitted Gaussian state and calibration. One conditional Gaussian path explicitly fits with all 2,699 training labels; another loads prefit state marked as trained on 2,699 official rows. Our 30 already-used holdout prompts are within those 150. Filtering runtime input to the 120 development prompts cannot remove information already encoded in frozen states. These states cannot be used as clean evidence in our fixed fivefold OOF. A component gain needs supervised parts rebuilt within each outer fold's 96 training prompts, or an independently sourced prompt set proven unseen by every artifact. A private, no-score resource study remains separate from that leak-free evaluation and still needs attachment and runtime checks.

## Remaining evidence

The competition input schema, hashes for attachments without trusted checksums, two T4 GPUs, model loading, complete 4,000-row inference, peak memory and disk use, and completion within nine hours remain untested. The public 0.684 score and its exact scoring run have not been independently verified. The CPU preflight is therefore a gate for further controlled testing, not evidence that the complete method is reproducible or competitive in our setup.
