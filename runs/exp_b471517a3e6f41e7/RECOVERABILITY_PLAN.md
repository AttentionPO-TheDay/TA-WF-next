# Frozen global representation recoverability pre-registration

本文件在任何 Day14/Day90/Day270 线性头拟合、特征评分或 query 成绩读取前冻结。TemporalDrift 是已观察的开发数据；本实验是有监督诊断，不是无标签方法、性能上界或新方法贡献。

## Provenance、表示与零训练边界

唯一 backbone 是 `exp_9121b664a1854097` 的正式 source-only best checkpoint：

| Backbone | Checkpoint | SHA-256 | Best epoch | 模型定义 | 最终 global embedding |
|---|---|---|---:|---|---|
| DF | `runs/exp_9121b664a1854097/checkpoints/df_best.pt` | `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae` | 29 | `src/ta_wf_next/models/df.py::DF` | `forward_features` / `forward` 的第二个返回值，即 `classifier` 最后一个 dropout 的输出、`mlp` 输入，512 维 |
| VarCNNDirection | `runs/exp_9121b664a1854097/checkpoints/varcnn_direction_best.pt` | `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83` | 23 | `src/ta_wf_next/models/varcnn.py::VarCNNDirection`，方向单分支派生，不称为完整 VarCNN | `forward_features` / `forward` 的第二个返回值，即 512→512 classifier 最后一个 dropout 的输出、`mlp` 输入，512 维 |

模型固定为 `eval()`，因此 dropout 关闭且 BN 使用 checkpoint 统计。输入为有符号时间戳前 5000 位取 `sign`，形状 `[B,1,5000]`。A 将未归一化的冻结 embedding 输入原 `mlp`；G-source、G-current 和 simple-maintenance 使用同一个逐行 L2 归一化 embedding，零向量保持零。禁止 optimizer/backward、backbone 微调、表示更新或 checkpoint 写入；新增 backbone 训练次数必须为 0。

正式 v3 split 为 `runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`，SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`。source supervised_train/reference/source-holdout 分别为 16,309/204/2,040，official `valid.npz` 全体是 source validation。模型与数据文件哈希必须在执行入口中复核。

## 四种读出数学定义

令冻结 embedding 为 `h(x) in R^512`，`z(x)=h(x)/max(||h(x)||_2, eps)`。

- A：原 checkpoint 分类头，`argmax_c (W_A h(x)+b_A)_c`。不使用 current support 更新，但在每个 date×shot×seed 的相同剩余 query 上重算。
- G-source：只用 v3 `supervised_train` 的 `(z,y)` 拟合带截距的 L2 multinomial logistic regression；仅用 official source validation 选择 C。reference、source-holdout、current support/query 均不拟合、不选模。
- G-current：从头初始化与 G-source 完全相同形式的 multinomial logistic regression，仅用该配置 current support 的 `(z,y)` 拟合。其 C 固定为同 backbone 的 G-source source-validation 胜出值；不从 query 选择 C、步数或早停，也不混入 source 样本。
- simple-maintenance：每类对 current support 的归一化 embedding 求均值并再次 L2 归一化为 `p_c`，预测 `argmax_c z(x)^T p_c`。不用 source prototype，无插值系数，无候选规则。

## 线性 solver 与冻结选择

G-source 与 G-current 均使用 scikit-learn `LogisticRegression(penalty="l2", solver="lbfgs", multi_class` 采用当前库的 multinomial 默认、`max_iter=300, tol=1e-4, fit_intercept=True, class_weight=None, random_state=6238)`。G-source 的唯一候选为 `C in {0.1,1,10}`，按 source-validation macro-F1 最大选择，完全并列取更小 C。每个 backbone 独立选择一次。G-current 直接继承这个 C；不做 current-side 超参搜索。记录 `n_iter` 和 convergence warning；若 300 次未收敛仍保留预注册结果，不增加迭代或据 query 改 solver。

## 日期、support 和共同 query

仅使用 Day14、Day90、Day270。固定 support seeds 为 1729、6238、20260916，预算为每类 3 和 10。对每个日期先执行下述内容级 canonicalization，再在每类 canonical row index 的升序数组上使用 `numpy.random.Generator(PCG64(seed)).permutation`；前 10 条是 10-shot support，前 3 条是 3-shot support，因此 3-shot 严格嵌套于 10-shot。该 shot/seed 的 query 是该日期全部 canonical 且未被排除的行减去对应 support。A、G-source、G-current、simple-maintenance 以及两个 backbone 都使用完全相同的 row ID 清单。若任一类不足 10 条，记录类别及上限并停止该日期全部正式配置，不删类、不降预算。

## 内容隔离审计与预定处理

admitted-input 内容定义为 `int8(sign(X[row,:5000]))` 的 5000 bytes，内容键为其 SHA-256；样本 ID 是文件名与原 row index。任何拟合/评分前完成并保存 `artifacts/data_isolation_audit.json` 与 `artifacts/support_manifests.json`：

1. 核验所有 NPZ、split 和 checkpoint 哈希；检查 source 三角色索引及 admitted-input 互斥，并纳入 official validation 全体。
2. 对每个 current 日期检查标签/类别计数、同内容组及冲突标签。若同一 current 日期内同内容有冲突标签则停止；否则每组只保留最小 row index，其余作为 current 内重复排除。
3. current 中任何 admitted-input hash 若出现在 source supervised_train、source validation、reference 或 source-holdout，整组 current 行全部排除，并按 source role、current 标签和 row ID 报告；不得跨角色保留。
4. 另外报告三个 current 日期间的内容交集，但日期各自独立评价，不因另一个 current 日期出现而删掉；本实验没有跨日期拟合或记忆。
5. 对每个 manifest 验证 support/query row index 与内容 hash 交集为 0，3-shot 是对应 10-shot 的严格子集，query 等于 eligible canonical pool 减 support，并保存 row-index 与 content-hash 清单及其 SHA-256。

审计失败或去重后任一类不足 10 条即停止受影响日期，不查看其 query 指标。不能静默保留重复、删类或改变预算。

## 指标、恢复量与预先判断门槛

每个 backbone×date×shot×seed×method 保存预测 artifact，并报告 accuracy、macro precision/recall/F1、102 类逐网站 accuracy。汇总 seed 均值和样本标准差（`ddof=1`）。主要恢复量均在完全相同 query 上逐配置计算：`G-current-A`、`G-current-G-source`、`simple-A`、`simple-G-current`；另对每个 backbone/date/method 报告 10-shot 与 3-shot seed 对齐后的差值。百分点差同时报告 accuracy 与 macro-F1，解释以 macro-F1 为主。

预注册的描述性 effect-size 门槛（“稳定/接近”不是显著性检验）：

- 输出端稳定可恢复：对每个 backbone，至少 2/3 日期的三个 seed 均有 `G-current-A > 0`，且该日期 seed-mean macro-F1 提升至少 5.0 pp；两个 backbone 都满足才称为跨 backbone 稳定恢复。
- 3-shot 是否足够：把上条对 3-shot 单独应用。若仅 10-shot 满足，则结论为需要 10-shot 级当前监督。10-shot 相对 3-shot 的 seed-mean macro-F1 增益在每个 backbone 至少 2/3 日期达到 2.0 pp，才称增加到 10-shot 有实质增益。
- simple-maintenance 接近 G-current：每个 backbone 至少 2/3 日期的 seed-mean `simple-G-current` macro-F1 绝对差不超过 2.0 pp，且跨全部日期/seed 的该 backbone mean 绝对差也不超过 2.0 pp。两个 backbone都满足才称为总体接近。
- 冻结表示路线依据较弱：若 10-shot 未达到“输出端稳定可恢复”门槛，且某 backbone 跨日期/seed 的 `G-current-A` mean macro-F1 提升不足 5.0 pp，则只报告有限恢复并建议另立匹配预算的表示更新实验；不宣称信息彻底消失。

无论结果如何，G-current 只说明少量当前监督下冻结表示仍有可利用信息；不得表述为表示天然抗漂移。结果出来后不修改本文件。实验完成即停止，不扩展日期、seed、模型、外部数据、DNNF/TFAN、TTA、多日期记忆或新机制。
