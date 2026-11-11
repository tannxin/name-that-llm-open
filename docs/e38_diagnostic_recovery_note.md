# E38 首次检索预检作废记录（2026-11-11）

E38 在分类评分前发现一次并发代码写入。首次 `diagnostic` 输出的 `diagnostic.json` 记录脚本 SHA-256 `d46327a63f0901356d0326cdff4a0b273f608afd2bc02ddc3bfc5117499a1601`、近邻映射 SHA-256 `6cf161a28e66d51c5c5cffc33db1d582d3f32078e84bfc217b27609703afc7d5`；运行期间另一个代理正在重排检索结果落盘与真实 prompt 审计的顺序。保存的映射包含新增的 `sparse_panel` 字段，但记录的脚本哈希对应过程中的文件状态，无法证实执行代码和磁盘代码完全同版。

`score` 命令在打开 E24 分数或写入 `score_started.json` 前被脚本哈希门槛拦截。首次产物完整保存在 Git 忽略目录 `outputs/e38_soft_prompt_neighbors_diagnostic_v0`，**不用于分类评分或晋级判断**。原预登记参数和门槛未根据首次检索结果修改。后续只允许在停止并发编辑、验证最终脚本哈希与语法后从空目录重新运行无标签预检；新的 `diagnostic.json` 与近邻映射哈希须匹配，才按预登记进入一次固定评分。
