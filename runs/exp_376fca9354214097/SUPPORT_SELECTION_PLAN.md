# Support-internal selection pre-registration

本文件在读取 `exp_b471517a3e6f41e7` 或 `exp_a2ffb5623ad346c7` 的 Day14/Day90/Day270 query 指标、结果正文，以及本实验任何 common-query 预测或评分前冻结。TemporalDrift 是已观察的开发数据；本轮只诊断普通 support-internal 选参，不是新算法贡献。结果出来后不修改本文件。

## 来源、表示与零训练边界

唯一 donor 是 `runs/exp_b471517a3e6f41e7`，历史固定规则对照来自 `runs/exp_a2ffb5623ad346c7`。唯一 backbone checkpoint 是 `runs/exp_9121b664a1854097/checkpoints/df_best.pt`（epoch 29，SHA-256 `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`）和 `varcnn_direction_best.pt`（epoch 23，SHA-256 `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83`）。正式 v3 split SHA-256 为 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`；support manifest、隔离审计、DF/VarCNNDirection G-source joblib SHA-256 依次为 `cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca`、`6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6`、`895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949`、`9e1d92c0fd5703473e4bbf4bd93e7cb6a7342100197a9e760a85b5c3d1fa777e`。入口必须复核这些哈希、donor integrity 内的 36 个正式 evaluation 哈希、数据文件哈希和 checkpoint history/metadata；任一不一致立即停止且不重建。

输入仍为有符号时间戳前 5000 位取 `sign`；模型固定 `eval()`；最终 global embedding 是原 `mlp` 输入的 512 维输出。G/prototype 方法只用逐行 L2 归一化 embedding。禁止 optimizer/backward、backbone 更新、微调与 checkpoint 写入；新增 backbone 训练和微调次数均为 0。

## support、fold 与 common query

日期固定为 Day14、Day90、Day270；support seeds 固定为 1729、6238、20260916；shot 固定为 3、10；类别固定为 102。只读解析 donor 的既有 manifest。不得重新抽样、换 seed、删类或改变 support 内容。

对每个 date×seed，common query 固定为该日期 canonical eligible pool 排除该 seed 的完整 10-shot support。该 common query 同时用于 3-shot 和 10-shot 的所有方法与两个 backbone。10-shot 使用每类全部 10 条 support；3-shot 只使用其嵌套的每类前 3 条做 CV 与最终重拟合，剩余 7 条既不训练、不选参，也不返回 3-shot query。程序须验证 3-shot support 是相应 10-shot support 的严格子集、common query 与完整 10-shot support 的 row/content hash 交集均为 0，并保存 common-query manifest 与 SHA-256。

CV 不增加随机性。对某配置，每类 support row 按 row index 升序排列，以类内位置 `j mod K` 指派 fold：10-shot 用 K=5，每 fold 每类验证 2 条、训练 8 条（816 train、204 validation）；3-shot 用 K=3，每 fold每类验证 1 条、训练 2 条（204 train、102 validation）。每个 candidate 在完全相同 folds 上评价。记录各 fold macro-F1/accuracy、均值及样本标准差（ddof=1）。候选拟合失败、缺类、非有限预测或 shrinkage 非有限优化均使该 candidate 在该配置标为 invalid，不换 fold、solver、预算或标签；若某 family 无有效候选则该 family 明确失败，不借 query 修复。

## 固定候选

类别数 K=102。数学定义完全沿用 `HISTORY_RETENTION_PLAN.md`：source/current prototype 都由归一化 embedding 的类均值再归一化；interpolation 混合后再归一化；shrinkage 从 donor `(W0,b0)` 初始化，使用相同 float64 SciPy L-BFGS-B、解析梯度、`maxiter=100,maxls=20,ftol=1e-9,gtol=1e-5`。G-current 沿用 donor 的 multinomial logistic regression 形式及 source-selected C=10，仅在相应 CV train fold 或完整 support 从头拟合。

候选一次性固定，不追加：

- prototype family：`G_source`（不更新端点，直接使用 donor 线性头）、`proto_alpha_0.25`、`proto_alpha_0.50`、`proto_alpha_0.75`、`current_prototype`（current-only 端点，等价 alpha=1）。
- linear family：`G_source`（不更新端点）、`shrink_lambda_1.00`、`shrink_lambda_0.10`、`shrink_lambda_0.01`、`G_current`（current-only 端点）。

`G_source` 是明确的不更新/G-source 端点；prototype family 中不把 source-prototype alpha=0 偷换成 G-source。上一轮固定 `alpha=0.25`、`lambda=0.1` 均保留。不存在更大网格、逐日期候选追加、额外 C、solver 或迭代选择。

## support-internal 选择、tie-break 与完整 support 重拟合

每个 backbone×date×shot×seed 独立选择。family 内按以下字典序排序有效候选：

1. fold mean macro-F1 较高；
2. 完全相等时 fold mean accuracy 较高；
3. 仍完全相等时 fold macro-F1 样本标准差较低；
4. 仍相等时采用预定保守顺序。prototype 为 `G_source, alpha=.25, .50, .75, current_prototype`；linear 为 `G_source, lambda=1.00, .10, .01, G_current`。

不设基于 query 的容差或规则。选定 candidate 后，使用该配置全部可用 support 重新构造 prototype 或重新拟合线性更新头；G-source 不拟合。随后才在 common query 固定预测。

最终 `support_selected_baseline` 也预注册：合并两个 family 的全部不重复 candidate，使用同一 CV 统计和上述前三项排序；最后 tie-break 的全局顺序固定为 `G_source, proto_alpha_.25, shrink_lambda_1.00, proto_alpha_.50, shrink_lambda_.10, proto_alpha_.75, shrink_lambda_.01, current_prototype, G_current`。这只是 support-CV 跨家族选择，不允许 query oracle 参与。family-specific selected 与跨家族 selected 都完整报告，不因某 family 的 query 结果选择展示对象。

程序对一个 backbone×date 必须先完成全部 shot×seed 的 CV、完整 support 重拟合及所有 candidate/common-query 预测并原子落盘；只有该单元全部预测固定后才可读取 common-query truth 评分。query 标签不得用于候选、fold、端点、family、early stopping、网站或任何预测决策。

## 对照、指标与汇总

每个 common-query 配置至少报告 `G_source`、`current_prototype`、`G_current`、固定 `alpha=.25`、固定 `lambda=.10`、support-selected prototype、support-selected linear 和跨家族 support-selected baseline。为公平比较，本 run 在新 common query 上重算 fixed-rule；它不覆盖也不改写旧实验结论。全部 candidate 也保存预测与 posthoc 指标，供预注册 oracle/归因使用。

每个方法报告 accuracy、macro precision/recall/F1、102 类指标；按 backbone×date×shot 对三个 seed 报均值/样本标准差。相对 G-source 逐样本报告 `corrected`、`harmed`、`net=corrected-harmed` 的计数及 query 比例，并汇总均值/标准差。每个配置保存所选 candidate、全部 CV fold 分数/方差、winner-runner margin、fold-wise winner 与优化状态；按 date/shot/backbone/seed 汇总端点与强度选择频率。

## 预注册判断门槛

主判断只用 10-shot 跨家族 `support_selected_baseline`，macro-F1 为主并同时报告 accuracy。

1. Day14 保护：每个 backbone 的 Day14 seed-mean macro-F1 不低于 G-source 超过 1.0 pp，且至少比同 shot 的两个 current-only 端点中较高者高 2.0 pp；两个 backbone 都满足才通过。
2. Day90/270 recovery：每个 backbone/date 定义 `R=max(G_current,current_prototype)-G_source`，`H=selected-G_source`。每个 `R>0` 单元均须 `H/R >= 0.80`；`R<=0` 单列。两个 backbone 的 Day90 和 Day270 全部通过才称保留大部分恢复。
3. 只有 1 与 2 同时通过，才称 10-shot 普通 support-internal 选择解决主要权衡，并把它作为强常规基线；本实验仍只报告证据，不替 Host 决定研究路线。3-shot 不纳入“足够”的主通过判定，只作低标签噪声诊断。

fixed-rule 到 selected-rule 的改善按相同 common query、同 backbone/date/shot/seed 配对报告：selected prototype 减固定 alpha=.25、selected linear 减固定 lambda=.10，以及跨家族 selected 减两固定规则中较高者，均给 accuracy/macro-F1 的 seed-mean 与总体均值。

## selection noise、oracle 与失败归因

posthoc query oracle 仅在全部 candidate 预测固定后计算：对每个配置以 common-query macro-F1 最大选择候选，完全并列沿用上述保守顺序。它只量化选择 regret 和候选 envelope，不作为最终性能。

对 3-shot 和 10-shot 分别报告：(a) selected 与 oracle candidate 一致率；(b) macro-F1 regret；(c) winner-runner CV margin；(d) selected candidate fold SD；(e) fold-wise winner 对最终 winner 的一致率；(f) seed 间选择频率。预先标记“高 CV 不稳定”为 selected fold SD >=5.0 pp，或 fold-wise winner 一致率 <60%，或 winner-runner mean macro-F1 margin 不大于两者 fold 均值差的标准误 `sqrt(sd1^2/K+sd2^2/K)`。预先标记“有实质 oracle 选错”为 selected 与 oracle 不同且 regret >1.0 pp。

失败归因按以下顺序，prototype、linear、跨家族分别量化，主结论以 10-shot 跨家族为准：

- A（selection failure）：posthoc oracle envelope 能通过上述 Day14+Day90/270 门槛，但 support-selected 未通过；或超过 1/3 的配置存在实质 oracle 选错，并同时显示上述任一 CV 不稳定信号。报告为 support 选择不可靠，不把 oracle 当性能。
- B（candidate-family failure）：即使把每个配置替换为 posthoc query oracle 后，oracle envelope 仍不能通过同一 Day14+Day90/270 门槛。此时候选族本身没有提供所需解；若同时存在明显 selection noise，报告 mixed，但不能把 B 隐去。
- 若 oracle 能通过而 selected 未通过但不满足高不稳定/高错配阈值，记为未被预注册诊断完全解释，不强行归为 B。若 selected 已通过，则不作失败归因。

3-shot 的同一统计专门解释低标签 selection noise；不增加标签、不跨日期合并、不使用 query 返调。

## 预算与停止条件

新增 backbone 训练/微调、checkpoint 输出均为 0。只允许冻结前向、CPU support-CV 与评分；不得加入 adapter、attention、memory、额外表示层、特殊 loss、无标签 TTA、外部数据、DNNF/TFAN、多日期连续记忆或 backbone update。完成固定 36 个配置、完整验证与 `RESULTS.md` 后停止，不追加日期、seed、候选或机制。
