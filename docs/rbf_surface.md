# E7b：RBF SVC 非线性表面特征诊断

## 实验设计

本实验复用 E0c 的词 1–2 gram、字符 3–5 gram TF-IDF 与 57 维风格特征。三个特征块的定义、标准化和归一化步骤保持不变，风格块权重固定为 1。每折都只用该折训练回答拟合 TF-IDF 词表和风格特征的 `StandardScaler`，再转换验证回答。分类器改为标准 scikit-learn `SVC(kernel="rbf", class_weight="balanced")`。

数据划分严格复用 E0：120 个开发 prompt、2,160 行，固定 5 折；30 个保留 prompt、539 行不进入特征拟合、模型训练、调参或评分。运行前逐行核对 E0c 的 ID、prompt、真实标签和折号，并用同一折的特征重新训练 E0c 的线性头，要求其 2,160 个折外预测与已保存的 E0c 预测完全一致。

网格事先固定为 `C∈{1,10}` 和 `gamma_multiplier∈{0.5,1,2}`，共 6 个候选。每一折用该折训练特征矩阵中固定随机种子抽出的最多 512 行，计算非自身配对的平方欧氏距离中位数 `d²_med`，再设 `gamma=gamma_multiplier/d²_med`。这样 gamma 尺度只由训练数据决定。每折的中位距离、抽样行数和 prompt 数写入 `fold_details.csv`。

使用五折合并的 OOF balanced accuracy 选择网格中的最高项，平分时选较小的 `C`、再选较小的 gamma multiplier。由于同一批开发折既用于选择又用于报告最优值，**最优 OOF 分数有选择乐观偏差**。它只能作为与 E0c 的探索性比较；最终性能要等整体方案确定后才在锁定的 30 个 prompt 上检查一次。

实现见 [`run_rbf_grid.py`](../run_rbf_grid.py)。输出目录为 `outputs/e7_rbf_svc/`，其中 `grid_results.csv` 保存全部候选的 OOF 和逐折分数，`best_oof_predictions.csv` 保存选中候选的逐行折外预测，`metrics.json` 记录训练文件哈希和选择说明。服务器另有只用开发集拟合的 `dev_model.joblib`；不必将模型权重加入 Git 仓库。

## 结果

| C | gamma multiplier | 合并 OOF balanced accuracy |
| ---: | ---: | ---: |
| 10 | 0.5 | **0.3185** |
| 10 | 1 | 0.3069 |
| 10 | 2 | 0.2875 |
| 1 | 1 | 0.2616 |
| 1 | 2 | 0.2606 |
| 1 | 0.5 | 0.2495 |

最高配置 `C=10, gamma=0.5/d²_med` 的五折分数分别为 0.3102、0.3264、0.3333、0.3032、0.3194；合并 OOF 为 **0.3185**。E0c 线性 SVC 在同一批折外回答上为 **0.3426**，高 0.0241。六个 RBF 候选均低于 E0c，故当前不将 RBF 分类器纳入后续方案。这个结论只针对这一组固定特征、缩放与有限网格；没有据此断言所有非线性核都无效。
