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

## Remaining evidence

The competition input schema, hashes for attachments without trusted checksums, two T4 GPUs, model loading, complete 4,000-row inference, peak memory and disk use, and completion within nine hours remain untested. The public 0.684 score and its exact scoring run have not been independently verified. The CPU preflight is therefore a gate for further controlled testing, not evidence that the complete method is reproducible or competitive in our setup.
