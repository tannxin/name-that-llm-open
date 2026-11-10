# E30 官方 Qwen3-Embedding-8B 镜像预检与获取通路

**状态（2026-11-10）：只完成两站元数据和小文件的核对；未下载权重、未运行 GPU、未训练 E30。**

E30-A 使用 [Qwen 官方 Hugging Face `Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af) 的不可变提交 `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`，镜像选择 [Qwen 官方 ModelScope 同名仓库](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B)。两站均标 Apache-2.0。该仓库是 **embedding 模型**，不能与 E28 的 `Qwen/Qwen3-8B` 生成模型或 Kaggle 的 `8b-base` 混用。

## 已完成的跨站核对

2026-11-10 分别读取 Hugging Face 上述**固定提交**的 tree API 和 ModelScope `master` 的仓库文件清单，逐项比较 9 个运行必需文件的精确字节数。四个权重分片及 `tokenizer.json` 的 Hugging Face LFS/Xet SHA-256 与 ModelScope `Sha256` 完全一致；四个较小文件从 Hugging Face 固定提交的 `resolve` 地址读取原始字节并计算 SHA-256，与 ModelScope 清单完全一致。

独立下载脚本的 `check --verify-hf` 已在本地实际运行通过，输出 9 个匹配文件、合计 15,146,107,275 字节，并明确标记 `weights_downloaded: false`。该检查不代表权重已在服务器落盘。

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

这些文件合计 **15,146,107,275 字节**。校验脚本会要求额外保留 10 GiB 可用空间。ModelScope 的 `master` 可移动，因此只能在实时清单再次通过上表的精确大小和 SHA-256 检查后下载。下载后每个文件再计算本地 SHA-256；重试及 Range 续传不会跳过校验。

## 运行步骤

[`download_e30_qwen3_embedding_mirror.py`](../download_e30_qwen3_embedding_mirror.py) 与 E28 下载脚本完全独立。下面的第一条命令仅请求 ModelScope 元数据；`--verify-hf` 还会请求 Hugging Face 固定提交元数据和 4 个小文件，总传输量低于 50 KiB。**只有显式 `download` 子命令会下载权重**。

```bash
python download_e30_qwen3_embedding_mirror.py check
python download_e30_qwen3_embedding_mirror.py check --verify-hf
python download_e30_qwen3_embedding_mirror.py download \
  --model-dir data/models/qwen3-embedding-8b-1d8ad4c --max-workers 3
```

训练服务器访问 Hugging Face 可能出现 DNS 失败；在这种情况下，`check --verify-hf` 无法完成实时跨站复核，但固定提交的两站字节比对已于 2026-11-10 在可访问环境完成。`download` 只要求 ModelScope 实时清单与已经跨站核对的固定值一致，因而不会因 Hugging Face 临时不可达而改用未审计权重。

模型目录位于 Git 忽略的 `data/` 下。获取完成后再运行 [`run_e30_supervised_embedding8b.py`](../run_e30_supervised_embedding8b.py) 的资源门槛；它会再次核对权重与 tokenizer 哈希，然后仅在 GPU 空闲时做单折资源测试。E28 下载正在进行时不要同时启动 E30 大文件下载，以免争用带宽和磁盘。
