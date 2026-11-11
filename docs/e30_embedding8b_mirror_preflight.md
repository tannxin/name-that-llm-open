# E30 官方 Qwen3-Embedding-8B 镜像预检与获取通路

**状态（2026-11-11）：固定提交的九个必需文件已下载并重新核对 SHA-256；E30-A 资源测试与官方开发集五折已完成。性能和部署边界见[五折独立审计](e30_embedding8b_fivefold_results.md)。** 下文保留下载前的预检与续传记录作为审计轨迹。

E30-A 使用 [Qwen 官方 Hugging Face `Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af) 的不可变提交 `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`，镜像选择 [Qwen 官方 ModelScope 同名仓库](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B) 的 Git 提交 `fcd9221ff0c460fdd81bfd04f4bc5b014e58da5b`。两站均标 Apache-2.0。该仓库是 **embedding 模型**，不能与 E28 的 `Qwen/Qwen3-8B` 生成模型或 Kaggle 的 `8b-base` 混用。

## 已完成的跨站核对

2026-11-10 分别读取 Hugging Face 上述**固定提交**的 tree API 和 ModelScope 上述**固定提交**的仓库文件清单，逐项比较 9 个运行必需文件的精确字节数。四个权重分片及 `tokenizer.json` 的 Hugging Face LFS/Xet SHA-256 与 ModelScope `Sha256` 完全一致；四个较小文件从 Hugging Face 固定提交的 `resolve` 地址读取原始字节并计算 SHA-256，与 ModelScope 清单完全一致。

ModelScope 提交由官方仓库的 `git ls-remote` 解析得出。已验证文件清单 `repo/files?Revision=<commit>` 和文件下载 `repo?Revision=<commit>&FilePath=config.json` 均接受该 40 位提交哈希，并返回表中一致的内容；不存在的 40 位哈希返回 HTTP 404，未静默回退到 `master`。ModelScope [仓库 API 文档/实现](https://github.com/modelscope/modelscope_hub/blob/main/src/modelscope_hub/api.py) 将 revision 描述为分支、标签或提交 SHA。下载器对清单和每个 Range 请求都使用同一个固定提交。

修订后的独立下载脚本 `check --verify-hf` 已在本地实际运行通过，输出两站固定提交、9 个匹配文件、合计 15,146,107,275 字节，并明确标记 `weights_downloaded: false`。该检查不代表权重已在服务器落盘。

服务器上也已运行不取权重的 `check`：固定 ModelScope 提交返回 9 个匹配文件，合计 15,146,107,275 字节，`weights_downloaded: false`。随后显式运行 `tokenizer-smoke`，只获取 `config.json`、`tokenizer_config.json`、`tokenizer.json`，逐项 SHA-256 通过；官方 tokenizer 实际追加的 `<|endoftext|>` ID 151643 在 attention mask 中为 1，符合 E30 末位池化规则。E28 的 13 个镜像文件全部下载和校验、独立哈希及 GPU 交接完成后，才启动 E30 的四片权重下载。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| `config.json` | 729 | `dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7` |
| `model.safetensors.index.json` | 30,432 | `ceaf1429bfa18eecc75d079737e8d1fe8dc0627f5ec7e6a985514626d1ac6a77` |
| `tokenizer.json` | 11,422,947 | `83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d` |
| `tokenizer_config.json` | 7,256 | `2f58f4bbd7bbce15d683f525954ef3a92cd82f5e06415a9c513859bf8ab72436` |
| `model-00001-of-00004.safetensors` | 4,900,037,024 | `99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451` |
| `model-00002-of-00004.safetensors` | 4,915,959,512 | `dff635b0f6dbbaad2a2d633ef037ec0a39bc165cc1806c712fbd6fcbcb4526c0` |
| `model-00003-of-00004.safetensors` | 4,983,067,656 | `30b1d4c53d84eb018f642cad7b373f0aabf79699872d8702c1f38577c0a59a2f` |
| `model-00004-of-00004.safetensors` | 335,570,376 | `36cbc9c60375693629f25743c1e77ebb1724af58e671b2376463193c7fd21ef6` |

这些文件合计 **15,146,107,275 字节**。校验脚本会要求额外保留 10 GiB 可用空间。虽然 ModelScope 的 `master` 可移动，下载器只查询和下载固定提交，且要求实时清单再次通过上表的精确大小和 SHA-256 检查。下载后每个文件再计算本地 SHA-256；重试及 Range 续传不会跳过校验。

若已有 `.part` 文件恰好达到预期长度但 SHA-256 错误，下载器会把它改名保留为 `.corrupt-<哈希前缀>-<随机值>`，重新核算剩余下载量和磁盘余量，再从零获取该文件。下载过程中发现校验错误，也会隔离 `.part` 并停止；下次运行重新执行空间检查后再下载。脚本不自动删除隔离文件。短读和网络异常使用带上限的指数退避，避免立即密集重试。

## 运行步骤

[`download_e30_qwen3_embedding_mirror.py`](../download_e30_qwen3_embedding_mirror.py) 与 E28 下载脚本完全独立。下面的第一条命令仅请求 ModelScope 元数据；`--verify-hf` 还会请求 Hugging Face 固定提交元数据和 4 个小文件，总传输量低于 50 KiB。`tokenizer-smoke` 只取得约 11 MB 的 tokenizer 和小配置并做 CPU 检查。**只有显式 `download` 子命令会下载权重**。

```bash
python download_e30_qwen3_embedding_mirror.py check
python download_e30_qwen3_embedding_mirror.py check --verify-hf
python download_e30_qwen3_embedding_mirror.py tokenizer-smoke
python download_e30_qwen3_embedding_mirror.py download \
  --model-dir data/models/qwen3-embedding-8b-1d8ad4c --max-workers 3
```

训练服务器访问 Hugging Face 可能出现 DNS 失败；在这种情况下，`check --verify-hf` 无法完成实时跨站复核，但固定提交的两站字节比对已于 2026-11-10 在可访问环境完成。`download` 只要求 ModelScope 实时清单与已经跨站核对的固定值一致，因而不会因 Hugging Face 临时不可达而改用未审计权重。

模型目录位于 Git 忽略的 `data/` 下。2026-11-10 09:07 UTC，后台下载 PID `1221761` 开始，当时已校验的 tokenizer 与小配置被跳过、剩余需传输约 15.135 GB。完整权重随后通过哈希核对；[`run_e30_supervised_embedding8b.py`](../run_e30_supervised_embedding8b.py) 的资源阶段再次核对模型文件，并在空闲 GPU 上通过门槛。五折汇总于 2026-11-10 23:40 UTC 完成，见[独立审计](e30_embedding8b_fivefold_results.md)。E30 下载本身没有使用 GPU。
