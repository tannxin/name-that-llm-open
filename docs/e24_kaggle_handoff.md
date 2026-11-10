# E24 Kaggle 离线候选交接

## 固定候选

E24 只替换 E16 的 E0c 线性分类器：其余全量 RoBERTa 和三个 ModernBERT seed、各模型 18 类分数的行内 z 标准化、ModernBERT 三种子均值及三路等权相加完全沿用 E16。候选的固定 120 prompt 五折 OOF balanced accuracy 为 **0.425000**，原 E16 为 **0.409259**（+0.015741）；4/5 折不降，最差类差值 −0.041667。五折原始结果及来源边界见 [E24 消融](e24_prvsiyan_external_ablation.md) 与 [来源审计](e24_prvsiyan_source_audit.md)。**0.425000 不是 Kaggle 分数。** 由于外部数据的生成协议参考过全部官方训练集的长度分布，这一结果仍是探索性的。

## 全量 E0c 拟合

在原服务器 Python 3.10.12 / scikit-learn 1.7.2 运行 [fit_e24_full_e0c.py](../fit_e24_full_e0c.py)。脚本重新验证官方训练文件 SHA `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`、原始外部 CSV、审计报告、4,322 行准入清单、E24 五折选出的 250 个 prompt 和 3,620 行清单。固定外部选样 SHA 为 `d7f8b7c9b0480ec896777e667589bf7d2fe12e503aa8f867c0708d0656ad6191`；没有看 OOF 或公榜重新选样。

全量训练为 **2,699 条官方回答 + 3,620 条审核通过的外部回答**，覆盖 17 个外部类别；Qwen 因实际服务快照与赛题目标关系未证实，外部权重 0。官方 `claude-fable-5` 为 **149 行**，其余各类为 **150 行**。有外部样本的每类总外部 `sample_weight` 均为官方该类行数的 **0.25**，即 Claude Fable **37.25**、其它覆盖类别 **37.5**；行权重为各自总量除以固定选中行数。原 E0c 的 TF-IDF、57 个风格特征、`LinearSVC(C=1, class_weight='balanced', dual=True, random_state=42)` 不变。拟合耗时 **40.320 秒**，模型 `e0c.joblib` SHA 为 `3826940c8ce46cb3ced1b8a72c0d0090f005fd3ea0596ff939567f9969d90ede`。这个全量模型无可报告的独立验证分数。

## 本地交付文件

| 文件 | 用途 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| `outputs/e24_kaggle_artifact_v1.zip` | 只含 E24 E0c 模型、元数据、三条 smoke 参考与拟合源码 | 20,696,249 | `cc73e5ec9187cd78c403d464b1ef1fd8d2b98f0873eb4a812eda335b3940d028` |
| `kaggle/submit_name_that_llm_e24.ipynb` | 独立 Kaggle Notebook | 46,069 | `838b50aef15806bba4aecf5acf00cc81b10c54de9a515200e2337a0a40bbbb77` |
| `kaggle/submit_e24_external_e0c.py` | Notebook 内嵌推理脚本 | 13,101 | `da06935ebfd5963c7e6c9a81dc5515bf60438d3cbe876db5325b02745912d8f2` |

小 ZIP 的五个内容文件与 manifest 在创建时逐文件及逐 ZIP 条目核验 SHA。它不含官方或外部回答文本、API envelope、训练/测试 CSV、其它神经权重；须保持私有。Notebook 重新核对新包、原 E0c/RoBERTa 包、三个 ModernBERT 包与源码哈希，并使用原离线 wheelhouse。外部来源在 Notebook 中署名为 [Prvsiyan Extended Name That LLM Dataset v9](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/version/9)，CC BY 4.0。

## 离线验证与提交配置

服务器原环境重载 E24 E0c 后，三条固定 smoke 文本的 18 类分数与训练时参考一致，argmax 不变。用官方 `data/test.csv` **3 行**运行完整 E24 脚本，原 RoBERTa 和三种子 ModernBERT 顺序完成，生成 Parquet；独立校验其仅有 `id,model` 两列、行数/ID 值与顺序精确一致且唯一、标签都属于官方 18 类。该样例提交 SHA 为 `70c8a7a9a003f0068682b2efc4e3260c0144845cccefa82903ca247437e8403a`。另外只对 E24 E0c 使用官方训练回答文本重复构成 4,000 行**无标签**夹具，18 类分数矩阵全有限，耗时 **14.733 秒**；这不是四模型端到端耗时。原 E16 四模型 4,000 行 CPU 验证耗时 83 分 24.582 秒，E24 额外 E0c 推理的数量级远小于神经模型，但 Kaggle 实际耗时未测。

Notebook 的全部代码单元已通过 Python 语法解析；新增 E24 挂载/manifest/源码核验单元用本地提取包实际执行通过。提交时应绑定**四个私有输入**：原 E0c/RoBERTa 模型包、E16 三种子模型包、原离线 wheelhouse、小 E24 ZIP；比赛数据由 Kaggle 自动绑定。Internet off、Accelerator None。Notebook 先验文件哈希，推理前再次校验 E24 类顺序、每类总权重与重载 smoke，随后生成 `/kaggle/working/submission.parquet`。若 Kaggle 页面显示与上述 SHA 不同，须先查明版本；不要以交互三行预览当作隐藏测试成功。

2026-11-06：已在 Kaggle 创建[私有 E24 模型数据集](https://www.kaggle.com/datasets/tanxin357/name-that-llm-e24-audited-external-e0c-artifact)，页面显示版本 1、20.7 MB、6 个文件。Notebook 尚未运行或提交，公榜/私榜分数未知。

2026-11-10：已在 [E24 Kaggle Notebook Version #1](https://www.kaggle.com/code/tanxin357/name-that-llm-e24-audited-external-e0c-submission?scriptVersionId=367930146) 核对 7 个代码单元及全部四个输入（比赛数据、原 E0c/RoBERTa、E16 ModernBERT、E24 模型数据集），会话设置显示 Accelerator `None`、Internet `off`。以 `E24 audited external E0c` 保存版本后，版本页显示 `Private`、`Running for 123.6s`，活动面板也显示 `Running`。

同日后续：Version #1 在 Kaggle 上 **165 秒成功完成三行预览运行**。Notebook 日志核实原模型、E16、E24 产物哈希，离线 wheelhouse 35 个哈希，三个 ModernBERT seed 全部推理完成，输出 `submission.parquet` 共 3 行。随后在[比赛提交页](https://www.kaggle.com/competitions/name-that-llm/submissions)选择此版本和该输出文件正式提交；提交列表出现 `Notebook Running` 记录及对应描述。Kaggle 正在以隐藏测试集私有重跑，公榜分数仍待出；三行预览的 165 秒不能外推为隐藏测试耗时。
