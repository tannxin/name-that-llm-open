# V63 公开高分 Notebook：资产与运行边界预审（2026-11-11）

本记录只审计公开元数据、源码和小型来源说明；没有下载模型权重、训练、提交 Kaggle 或使用锁定保留集。[V63 Notebook](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble?scriptVersionId=366688484) 是参考系统，不是本项目独立验证的候选。

## 固定来源与输入

2026-11-11 通过 Kaggle 公开 [`kernels/pull` API](https://www.kaggle.com/api/v1/kernels/pull?userName=lightningv08&kernelSlug=0-680-modernbert-lora-sparse-ensemble) 读取当前版本 **63** 的 `blob.source`；其 UTF-8 字节 SHA-256 为 `a7e2c322caf975d460fcf3544b175bec949a8dd0c8aae77c6d87d16978f8a819`。元数据标记 Notebook 公开、关闭互联网、使用 Tesla T4 GPU；没有返回第 63 版的代码许可证或隐藏重跑耗时。

元数据绑定 **21 个数据集、8 个 Notebook 输出、6 个 Kaggle 模型版本**。主要资产如下，大小为查询当天的公开数据卡数字，不能把不同版本混为同一文件：

| Kaggle 数据集 | 查询时版本 | 总大小 | 数据卡许可 |
| --- | ---: | ---: | --- |
| [主资产包](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) `lightningv08/ntl-olmo2-native24-final-inference-assets` | 22 | 104.724 GB | Other，组件分别授权 |
| `dhruvtibarewal/nlm-l24-adapters-v5` | 1 | 2.592 GB | Other |
| `nishantdahal/ntl-handwriting-blend` | 1 | 1.281 GB | CC BY-SA 4.0 |
| `josephadamski91/ntl-atlas-qwen3-8b-v1` | 3 | 0.463 GB | Other |
| `pjmathematician/ultrabert-380m-ntl` | 1 | 3.819 GB | Apache 2.0 |
| `bbucxi/qwen35-9b-safetensors` | 1 | 19.329 GB | 未标明 |

另有 `dhruvtibarewal/nlm-l24-adapters`、`seyitkaangunes/ntl-extra-api-passages`、`abdullah1afify/ntl-generations`、`dophuchung/ntl-proxy-generations`、`zbigniewkirchner/ntl-qwen3-8b-lora-claude-head`、`lucasboesen/ntl-free-proxy-generations`、`nuridudak/ntl-q14-lora`、`nuridudak/ntl-q14-lora-v4`、`kirilltym/ntl-spectral-style-heads-paired-4404`、`nuridudak/ntl-svc-prefit-pool-v4n`、`dhruvtibarewal/nlm-enc-v5`、`nuridudak/ntl-two-stage-encoders`、`nishantdahal/ntl-two-stage-lora`、`mercurydust/ntllm-chatgpt-a-b`、`prvsiyan/prvsiyans-extended-name-that-llm-dataset`。六个模型版本是 Qwen3 的 4B/8B/14B Base、Qwen3-Embedding-8B、Qwen2.5 的 7B/14B；八个 Notebook 输出由 `lucasboesen` 的七个权重/训练 Notebook 和 `vnton37nguyn/ntl-train-lora-qwen25-instyle` 组成。具体绑定以固定源码元数据为准；有些附件可能只是历史或可选依赖，不能把总挂载大小当作实际峰值内存。

主资产包数据卡明确其 **Other** 类别不重新授权组件：其中有 Apache 2.0、CC BY-SA 4.0、BSD、PriorLabs 非商业条款，`llama31-prefix24` 另受 Llama 3.1 Community License 约束。V63 镜像中的 `ntl-native-q8-multidepth-fivefold-0679` 五折状态约 0.93 GB；其小型 `ORIGIN.json` 指向 CHEN Han 的 `concyclics/ntl-native-q8-multidepth-fivefold-20261103`，原始数据集查询返回 403。镜像保留来源说明，不等于获得将全部权重重新发布为 MIT 的许可。第 63 版 Notebook 代码许可仍需在 Kaggle 页面单独核实；源代码许可也不覆盖挂载权重。

## 实际运行路径和验证边界

第 63 版源码的核心成员是 Qwen3-14B、Qwen3-8B、Llama 8B、另一 Qwen3-14B、Qwen2.5-14B 和 DeBERTa large；执行代码明确断言核心成员中没有 `modernbert`，因此标题不能用作架构清单。核心之后还依次运行互补模型、低间隔行的条件 ATLAS、27B GSQ、五折 Qwen3-Embedding-8B，再融合输出。后者要求两张 GPU，其五折按 512-token 覆盖窗口逐次前向。主源码记录的独立核心基座文件量为 Qwen3-14B 29.55 GB、Qwen3-8B 16.39 GB、Qwen2.5-14B 29.55 GB、Llama 前 24 层 11.53 GB；这些是挂载/暂存大小，不是峰值 GPU 用量。

源码对三个固定公开样例核对 ID、字节长度和文本指纹，命中便只写格式预览，跳过所有 GPU 前向。公开页约 21 秒的预览时间**不能证明**约 4,000 条隐藏测试的九小时推理与资源门槛。公开 0.680 分数可作为该 Notebook 的外部成绩记录；本次未取得该版本完整隐藏重跑日志或逐阶段耗时。公开适配权重可能已训练于全部官方训练行，不能用本项目现有 120-prompt 五折 OOF 估计其无泄漏增量。

**下一检查点：**固定 V63 源码与主资产包 v22，先逐项核对附件可访问、实际读取文件的大小/哈希和组件许可；再用同等双 T4、约 4,000 条无标签替身文本做完整路径的逐阶段计时与峰值内存试验。该检查不使用榜单或保留集调权重，不复制第三方权重到本项目公开 MIT 仓库，也不把参考系统的公榜分数当作我们方案的离线成绩。
