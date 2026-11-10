# E16 Kaggle 离线候选交接

## 固定方法与当前证据

保留已经成功提交的 E0c + RoBERTa 全量模型包，只新增三个全量 ModernBERT-base 模型（seed 42、17、73）。每条回答的每一路 18 类分数分别做行内 z 标准化；三个 ModernBERT 标准化分数先取均值成为一路，再与 E0c、RoBERTa 两路等权相加。这个规则在开发 prompt 的预定五折 OOF 上为 **0.409259** balanced accuracy；原 E0c + RoBERTa 为 **0.399537**。用提交脚本的标准化函数和融合算式重算 2,160 条开发 OOF，逐行预测与保存的 E16 预定集成结果 **全部一致**。原 v3 Kaggle 公榜为 **0.401**，E16 已成功取得公榜 **0.407**。已锁定的 holdout 没有用于 E16 权重、种子或提交规则选择。

## 可上传文件与核验

| 用途 | 本地文件 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| 原 v3 E0c + RoBERTa 私有 Dataset | `outputs/kaggle_artifact.zip` | 520,528,073 | `b261a657b077ae790072e5ee90cd2db88f749653e57135efe996471848b6b1dc` |
| 原 v3 离线 wheelhouse 私有 Dataset | `outputs/kaggle_wheelhouse_cp313_linux_x86_64.zip` | 328,485,282 | `2c4dc134de5717ed8d974c4fb8405f44d848045c08b08706826436ec66480a84` |
| 新 E16 三 seed 私有 Dataset | `outputs/e16_kaggle_artifact_v2.zip` | **1,799,379,838** | **`48c0043f7266296284de349e0b83d7e041f11efcf50772d5e65917ccdb781e99`** |
| 新 E16 Notebook | `kaggle/submit_name_that_llm_e16.ipynb` | 见文件 | `98156d06e9b946af43542a486e6e48dc966ebaf6ae6efca177f29f6c9975313c` |

新 ZIP 有 26 条目（manifest 和 25 个文件），只含三个模型包及原样的 `fit_modernbert_full.py`，不含训练/测试 CSV。打包前逐包核对了 revision、官方 18 类顺序、2699 行/150 prompt、训练数据 SHA、训练设置、运行时、源码 SHA、权重及 tokenizer/config 文件哈希；三个 `smoke_verification.json` 均报告 CPU 重载 argmax 一致和绝对 logits 差小于 `2e-5`。打包后又逐 ZIP 条目核对 SHA，独立 `Get-FileHash` 与打包器报告一致。Notebook 会在推理前重新逐文件核对两个模型 Dataset；当运行时需要离线 wheelhouse 时，还会核对其全部 35 个 wheel。推理脚本再次验证每个模型及三条 smoke 文本；跨 CPU 构建复算的 logits 容差预设为绝对 `1e-3`，同时要求 smoke argmax 完全一致，包内原机 `2e-5` 门槛和文件哈希不变。

训练/加载源 `fit_modernbert_full.py` 的固定 SHA-256 为 `2653db83407da1531598f80e0d02b4d5c958cc086048a1a540b485146aa112e8`；Notebook 内嵌的提交源 `submit_e16_three_seed.py` 的 SHA-256 为 `c36b864e55f3e0f1fbc025b5934cf21367fb8ae5b65e85b2393e2668815aa20b`。

## 性能验证

服务器 CPU、4 线程、batch 16 的完整 **4,000 行离线端到端测试通过**，wall time **83 分 24.582 秒**。E0c + RoBERTa 阶段约 18 分 41 秒结束，ModernBERT seed42 和 seed17 分别在总 elapsed 约 42 分和 64 分时结束，seed73 随后完成；峰值观察到的常驻内存约 4.0 GB。输出 `outputs/e16_cpu_4000_submission.parquet` 的 SHA-256 为 `0ccca919e04037b60d9f7a1374d7f2cc57da687e35782b9b3e7b78827811dfed`。独立读取 Parquet 验证：恰有 `id,model` 两列、4,000 行、ID 与输入值及顺序完全一致且唯一、无缺失标签、所有预测均属于官方 18 类。fixture 只从 120 个开发 prompt 的文本抽样，**从未读取标签列**；fixture SHA-256 为 `0b71b8ed1807dfef7bb9802245445990c45d84dfdd50999c4a3ae438df33977b`。实际 Kaggle CPU 耗时可能不同。

上述 4,000 行测试启动时采用原机较严的 `2e-5` smoke 容差。随后用当前固定提交脚本（SHA-256 `c36b864e55f3e0f1fbc025b5934cf21367fb8ae5b65e85b2393e2668815aa20b`，跨运行时容差 `1e-3` 并要求 argmax 一致）对官方 3 行样例再次运行，**38.308 秒成功**；输出 schema、ID 顺序和标签校验通过。离线 wheelhouse 已静态核对含 CPython 3.13 Linux x86_64 的 `torch 2.6.0+cpu`、`transformers 4.51.3`、tokenizers、safetensors 等 35 个 wheel，Transformers wheel 内含 ModernBERT 实现，三个 E16 包均含本地 config/tokenizer。Kaggle 交互运行中已实际完成离线 wheelhouse 安装、`torch 2.6.0+cpu` 和 ModernBERT 配置/分词器探测，三行预览生成了有效 `submission.parquet`。

## Kaggle 实施记录（2026-11-06）

1. 已上传 1,799,379,838 字节 ZIP 并创建[私有 E16 Dataset](https://www.kaggle.com/datasets/tanxin357/name-that-llm-e16-modernbert-three-seed-artifacts)，页面可见 26 个解包文件。
2. 已创建[私有 E16 Notebook](https://www.kaggle.com/code/tanxin357/name-that-llm-e16-three-seed-modernbert-submission)，绑定比赛数据、原 E0c/RoBERTa 模型与 wheelhouse、新 E16 Dataset；Internet off、Accelerator None。三行交互运行完成离线依赖安装、模型文件校验和提交文件检查。
3. 已保存并运行 Notebook Version #1（显示耗时 2 分 46 秒），生成预览 `submission.parquet`。2026-11-06 **12:37:41 北京时间**点击 Submit to Competition，Kaggle 接受提交并对隐藏测试重跑。随后[提交列表](https://www.kaggle.com/competitions/name-that-llm/submissions)显示 **Succeeded，公榜 balanced accuracy 0.407**；原 E10 为 0.401，同一公榜指标下高 0.006。E16 提交所用的 [Notebook Version #1](https://www.kaggle.com/code/tanxin357/name-that-llm-e16-three-seed-modernbert-submission?scriptVersionId=366158472) 保持锁定。

交互运行只有三行公开样例；其成功确认依赖和接口可用。隐藏重跑现已成功，公榜结果单独记于上文；私榜分数尚未知。
