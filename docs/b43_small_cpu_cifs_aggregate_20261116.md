# B43 small-file CPU and CIFS preflight (2026-11-16)

Status: **small-file CPU preflight passed for four eligible candidates; large-weight transfer and GPU remain NO-GO**. This check used no train/validation labels, model weight shards, or GPU.

## Exact public model sources

| Candidate | Official repository | Immutable revision | Small files checked | Indexed weight shards | CPU result |
| --- | --- | --- | ---: | ---: | --- |
| M7 | `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | 7 | 3 | Fast tokenizer and 18-class meta head pass with isolated optional dependencies |
| M12 | `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f977929a969745b69bbd62e03043551a5` | 8 | 5 | Pass |
| Q7 | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` | 6 | 4 | Pass |
| Q14 | `Qwen/Qwen2.5-14B` | `97e1e76335b7017d8f67c08a19d103c0504298c9` | 6 | 8 | Pass |

The 27 config/index/tokenizer files total 56,223,274 bytes. Every file was checked against its fixed revision's official Git blob OID or LFS SHA-256, exact byte size, and a prior sampled SHA-256 when available. Each downloaded safetensors index references exactly the shard set in the independently captured official catalog. The actual weight shards were **not downloaded or hashed**. The original six-candidate statistical registry remains intact: the two OLMo 2 candidates are still ineligible under the pinned classification protocol.

The restored CPU environment used PyTorch `2.6.0+cu118`, Transformers `4.51.3`, `trust_remote_code=False`, offline mode, and `CUDA_VISIBLE_DEVICES=`. Config and default fast tokenizer loading, three synthetic text tokenizations, `AutoModelForSequenceClassification` mapping, and empty meta-tensor classification head shapes passed as follows: M7 `[18,4096]`, M12 `[18,5120]`, Q7 `[18,3584]`, Q14 `[18,5120]`. Qwen 2.5 emitted a generic SDPA sliding-window warning during meta construction; the pinned configs set `use_sliding_window=false`. This does not replace a weighted forward or resource test.

M7 default `AutoTokenizer` failed in the restored runtime because `sentencepiece` and `protobuf` were absent. It passed after adding only `sentencepiece==0.2.0` and `protobuf==5.29.5` in an isolated CPU-only dependency overlay; the original runtime was not modified. The Linux wheels matched the SHA-256 on their official PyPI release pages: `1380ce6540a368de2ef6d7e6ba14ba8f3258df650d39ba7d833b79ee68a52040` and `63848923da3325e1bf7e9003d680ce6e14b07e55d0473253a690c3a8b8fd6e61`. A future M7 execution recipe must explicitly lock and package these dependencies, including the Kaggle offline environment.

## Live CIFS publication check

An isolated one-MiB file was created and then cleaned up. File `fsync` passed; directory `fsync` passed before and after publish; `renameat2(RENAME_NOREPLACE)` returned `EEXIST` without changing an existing target, then published to a new name and preserved SHA-256. `os.link` hard-link publication failed with `ENOTSUP` (errno 95). An earlier hard-link publication path cannot run on this PVC. The newer no-replace rename draft remains review-only until fault injection and independent review.

## Remaining execution gates

Current authoritative Pod and node storage telemetry is unavailable, so local temporary-storage use, node free space, and eviction thresholds cannot be continuously attested. Dashboard Ready/absence of DiskPressure, Pod `df`, and ample shared-PVC space do not satisfy the pre-registered Stage 1 safety gate. The source registry and full-candidate downloader remain review-only. A separate PVC-only, one-shard pilot draft is disabled pending fault injection and independent review. No large model transfer or GPU run was started.

Local verification report SHA-256: `5ce2e604f69af18b0a5a801b76a2c94bfbfd149f204f3a0f2945cd930b7e03ae`.
