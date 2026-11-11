# E37：CPU 来源条件四字符证据的预登记

**在任何 E37 验证分数出现前锁定。** 本实验只测 ATLAS 所用“逐局部单元比较并截断相对似然”机制的低成本计数代理；它不估计因果 LM 的逐 token 条件概率，也不是 ATLAS 复现或其收益估计。算法依据 [ATLAS 原论文 §3.2 与 Appendix B.1](https://arxiv.org/html/2610.19583v1)，但代码独立实现，不复制作者仓库源码。作者的五家族模型与本赛 18 个精确版本不同。[方法边界说明](p2_likelihood_atlas_feasibility_2026-11-11.md)

## 数据与固定五折

只用官方 `data/train.csv` SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`。用 E0 `split_manifest.csv` SHA-256 `653c23b191b40086f7a0263a9baee1e519c0778de7bd0c721b23e1a9955ffd15` 和 `oof_predictions.csv` SHA-256 `698265e92c50f23acc6eb68731721249a27a5966e30b62c4eed3c0a30c3528fa` 恢复原 120 prompt、2,160 行开发集和五个 24-prompt 验证折。每折只从另外 96 prompt、1,728 行（18 类各 96 行）拟合词表、计数和分母；验证折 432 行（18 类各 24 行）仅在拟合结束后评分。30 prompt 留出集只在原始 CSV 的全文件 SHA 核对中被动经过，文本不进入 E37 数据结构；`test.csv` 不读取。无外部训练数据。

## 唯一固定模型与两个预设读出

原始 Unicode 文本不做归一化或大小写变换；每个连续四字符片段 $g_t$ 是一个观察单元，重叠片段都计数。折内训练文本形成至多 100,000 个最高频片段的词表，词表外统一映射 OOV。对每个来源类 $k$，设 $n_{k,g}$ 为训练侧片段频数、$n_{k,\mathrm{OOV}}$ 为词表上限排除的训练片段次数、$N_k=\sum_{g\in V}n_{k,g}+n_{k,\mathrm{OOV}}$、$V$ 为训练侧词表大小；加一平滑且 18 类先验相等：

$$p(g\mid k)=\frac{n_{k,g}+1}{N_k+V+1}\quad(g\in V),\qquad p(\mathrm{OOV}\mid k)=\frac{n_{k,\mathrm{OOV}}+1}{N_k+V+1}.$$

被词表上限排除的训练片段视为 OOV，它们的数量纳入 $N_k$；因此 $N_k$ 是该类**全部**训练片段发生次数，而非仅词表内次数。上述 $V+1$ 个概率严格归一。令 $\ell_{k,t}=\log p(g_t\mid k)$。主读出严格用论文的相对似然截断式：

$$r_{k,t}^{(T)}=\exp\!\left[-\frac{1}{T}\sum_{j\ne k}\max(\ell_{j,t}-\ell_{k,t},0)\right],\qquad T\in\{1,2,4,8\},$$

$$R_k=\frac{1}{4}\sum_{T\in\{1,2,4,8\}}\frac{1}{m}\sum_{t=1}^{m}r_{k,t}^{(T)},\qquad \hat k=\arg\max_k R_k.$$

$m=\mathrm{len}(text)-3$，包括词表外片段。重复的片段可按出现次数加权精确合并。次读出固定为 $S_k=m^{-1}\sum_t\ell_{k,t}$；只作机制诊断，不参与晋级或融合选优。18 类标签顺序沿用 `run_e0.LABELS`，`argmax` 并列按该顺序。任何非有限值、零片段或折内类别缺失立即停止。

## 评估、资源和晋级边界

先在 CPU 运行 `lock`：核对数据、开发集和五折不变量、旧 E24 源文件的 SHA、代码/配置/本文 SHA、Python/NumPy/pandas/scikit-learn 版本；只生成不可覆盖的 `outputs/e37_char4_likelihood/protocol_lock.json`，不拟合、不读 E37 验证分数，也不读旧 E24 分数内容。锁定后 `run` 仅允许运行一次，脚本与配置 SHA 改变即拒绝。完成全部五折之前不保存、不显示任何单折验证分数；五折完整后才写 Git 忽略的逐行 OOF 和聚合报告。

比较：主读出单路；E24 固定三路融合加上 `1.0 × row-z(R)`，不改变 E24 原分支、类别顺序或系数。先从锁定的 E24 OOF 源逐行重现旧 `pred_e16_with_e24` 和 balanced accuracy `0.425000`，再看 E37 增量。报告总体 balanced accuracy、五折、18 类召回及变化、120 prompt 配对 bootstrap 20,000 次 seed 42 的区间。主融合相对 E24 须 **BA 至少 +0.01、至少 4/5 折不降、任一类召回下降严格小于 0.05**，才建议进入神经来源条件模型的独立资源预检；这不是完整 ATLAS 或 Kaggle 晋级门槛。次读出不得替代主读出。不根据单折、保留集或公榜调参。

资源门槛：只用 CPU；每条文本最多 10,000 字符；峰值进程 RSS **不超过 4 GiB**、每折 **不超过 900 秒**、全程 **不超过 3,600 秒**。超过即失败，不改词表、平滑、温度或数据范围补救；脚本保存失败原因而不保存部分验证分数。没有模型权重或数据进入 Git。作者 [ATLAS 论文](https://arxiv.org/html/2610.19583v1)的 7B 主实验用 4 张 H100 和远多于本赛每折每类 96 条的训练数据；本 E37 结果不能外推为神经 ATLAS 的显存、时间或效果。
