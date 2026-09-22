# Historical retention baseline pre-registration

本文件在本 experiment 的任何 Day14/Day90/Day270 query 新预测或评分前冻结。TemporalDrift 是已观察的开发数据；prototype interpolation 与 shrink-to-source linear update 都是常规历史保留基线，不是新方法或创新。已有 donor 结果是本实验动机和必须复核的对照，不用于选择下述 alpha/lambda。

## Provenance、表示与零训练边界

唯一 donor 是 `runs/exp_b471517a3e6f41e7`，唯一 backbone checkpoint 是 `runs/exp_9121b664a1854097/checkpoints/df_best.pt`（epoch 29，SHA-256 `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`）与 `varcnn_direction_best.pt`（epoch 23，SHA-256 `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83`）。模型固定 `eval()`，dropout 关闭，BN 使用 checkpoint 统计。输入为有符号时间戳前 5000 位取 `sign`，形状 `[B,1,5000]`。最终 global embedding 是 `forward_features`/`forward` 的第二返回值、原 `mlp` 输入，512 维；所有 G/prototype/history 方法使用逐行 L2 归一化的 `z=h/max(||h||_2,eps)`，零向量保持零。

正式 source split 是 `runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`，SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`。固定 current manifest 是 donor `artifacts/support_manifests.json`，SHA-256 `cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca`；隔离审计 SHA-256 是 `6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6`。两套 G-source joblib SHA-256 分别为 DF `895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949`、VarCNNDirection `9e1d92c0fd5703473e4bbf4bd93e7cb6a7342100197a9e760a85b5c3d1fa777e`。

程序入口必须在 source-side 选择和正式评价前逐项复核以上哈希、donor 36 个评价 artifact 的 integrity-check 哈希、manifest 全部 18 个配置及 checkpoint metadata/history。发现不一致立即停止，不重建 manifest、不重做隔离审计。禁止 optimizer/backward、backbone 参数更新和 checkpoint 写入；新增 backbone 训练与微调次数均为 0。

## 固定方法定义

类别数 `K=102`。所有方法在每个 backbone×date×shot×seed 的同一 manifest query 上比较。

- `G_source`：直接复用 donor source-only multinomial logistic head。其 `C=10` 已由 donor 仅在 official source validation 选定，不使用 current support。
- `current_prototype`：即 donor `simple_maintenance`。对该配置 support 的归一化 embedding 按类别求均值，再逐原型 L2 归一化，按 query 与原型余弦最大预测。
- `G_current`：直接复核 donor 对照；从头、仅用该配置 support 拟合与 G-source 同形式且继承 `C=10` 的 multinomial logistic regression。
- `prototype_interpolation`：source prototype `s_c=normalize(mean_{i in source supervised_train,y_i=c} z_i)`；current prototype `u_c=normalize(mean_{i in current support,y_i=c} z_i)`；固定候选 alpha 下 `q_c(alpha)=normalize((1-alpha)s_c+alpha u_c)`，预测 `argmax_c z(x)^T q_c(alpha)`。所有均值先在已逐行归一化 embedding 上计算；混合后再次归一化。
- `shrink_to_source`：令 donor G-source 参数为 `(W0,b0)`，从该参数初始化，用 current support 最小化

  `J(W,b)=-(1/n) sum_i log softmax(W z_i+b)_{y_i} + (lambda/2)(||W-W0||_F^2+||b-b0||_2^2)`。

  使用 SciPy L-BFGS-B、解析梯度、float64 参数与目标，`maxiter=100`、`maxls=20`、`ftol=1e-9`、`gtol=1e-5`；没有 validation early stopping。达到迭代上限仍保存预注册终点并记录状态，不增加迭代、不换 solver。预测为 `argmax_c(Wz+b)_c`。

原 checkpoint 分类头 A 可从 donor 保留作上下文，不参与 alpha/lambda 选择，也不是本轮核心机制。

## alpha/lambda 的 source-only 共享选择

小规模候选固定为 `alpha in {0.25,0.50,0.75}`、`lambda in {0.01,0.10,1.00}`，不增加候选。

选择只使用 source side。对每个 backbone，在 v3 `source_holdout` 内按类别升序行数组和 `numpy.random.Generator(PCG64(seed)).permutation`，用 seeds 1729/6238/20260916 取每类前 10 条，前 3 条严格嵌套，形成 3/10-shot 伪 current support。source prototype 始终只由 `supervised_train` 构造；每个候选用伪 support 构造/拟合，并在与这些角色内容隔离的 official source validation 上计算 macro-F1。分别对 alpha 和 lambda，在两个 backbone×两个 shot×三个 seed 的 12 个 macro-F1 上取未加权均值最高者；alpha 完全并列取较小值，lambda 完全并列取较大值。最终各得到一个跨日期、跨 shot、跨 seed、跨 backbone 共享的 alpha 和 lambda。选择记录必须保存各候选全部 12 个分数、均值/样本标准差、support 行哈希和最终值。

该规则不读取或使用 Day14/90/270 的任何 query 输入、标签、预测或指标；也不逐日期、逐 backbone 选择。source-side 选择完成并将 `source_selection.json` 原子落盘后，正式评价入口才允许运行。

## current manifests、标签权限与执行顺序

日期固定 Day14/Day90/Day270，shots 固定 3/10，seeds 固定 1729/6238/20260916。直接解析 donor manifest，禁止随机抽样、删类、换 query、改变 shot 定义或重做内容隔离审计。current support 标签仅供 current prototype、G-current、interpolation、shrinkage。对每个 backbone×date，程序先为全部六个 shot×seed 配置固定所有新方法预测；只有之后才访问 query truth 进行评分和 posthoc 分析。query 标签绝不进入 alpha/lambda、权重、迭代数、正则化、停止、模型选择、网站筛选或预测决策。

既有 G-source/current prototype/G-current/A 的 row、truth、预测与指标从 donor 只读 artifact 复核并复用；新方法不覆盖 donor。所有新增代码、selection、预测、指标与报告仅写入本 run。

## 输出、指标与错误转移

每个配置保存 query row/truth、六种方法预测（A 仅上下文）、accuracy、macro precision/recall/F1、102 类 precision/recall/F1/accuracy。核心汇总对五种方法 G-source、current prototype、G-current、prototype interpolation、shrink-to-source 报告三个 seed 的均值与样本标准差（ddof=1）。

对每个非 G-source 方法逐样本定义：`corrected = 1[G_source错且该方法对]`，`harmed = 1[G_source对且该方法错]`，`net=corrected-harmed`。按 backbone/date/shot/seed 报告计数及占 query 比例，再按 backbone/date/shot 汇总三个 seed 的均值/样本标准差。逐网站报告 corrected/harmed/net，并检查 net 符号是否在至少 2/3 seeds 重复、是否在至少 2/3 dates 重复、是否在两个 backbone 重复；这些仅是结果后的解释，不反馈到选择。

预测时可见诊断固定包括：G-source 与历史方法是否分歧、G-source top-1 margin、历史方法 top-1 margin、interpolation 的最大 prototype cosine，以及 support prototype 与 source prototype 的同类 cosine。评分后按正确/错误、corrected/harmed 分组报告这些量，并检查跨 backbone 重复；不据此改变预测。

## 预注册判断门槛

以下均为描述性 effect-size 门槛，不是显著性检验；以 macro-F1 为主，同时完整报告 accuracy，不能只展示有利日期。

1. **Day14 保护**：某历史方法在一个 backbone 上通过，当 Day14 的 3-shot 与 10-shot seed-mean macro-F1 均不低于 G-source 超过 1.0 pp，且均至少比同 shot 的两个 current-only 方法中较高者提升 2.0 pp。称跨 backbone 保护须两个 backbone 都通过。
2. **保留后期恢复**：对 Day90、Day270、每个 shot，定义 current-only 可用恢复 `R=max(F1_G_current,F1_current_prototype)-F1_G_source`，历史方法恢复 `H=F1_history-F1_G_source`。当 `R>0` 时保留率为 `H/R`。某 backbone 通过须其所有 `R>0` 的 Day90/270 单元保留率至少 0.80；两个 backbone 都通过才称跨 backbone 保留大部分后期恢复。`R<=0` 的单元单独报告，不用负分母制造有利比例。
3. **简单历史保留已经足够**：同一历史方法必须同时通过跨 backbone Day14 保护与跨 backbone后期恢复；否则结论为尚不足。若任一简单方法通过，把它列为后续强基线并停止复杂机制扩展。
4. **3-shot 瓶颈缓解**：先报告每个方法 seed 配对的 10-shot−3-shot macro-F1。历史方法只有在两个 backbone 各至少 2/3 日期把相对较强 current-only 方法的 shot gap 缩小至少 50%，且其 3-shot 相对 G-source 不低于 -1.0 pp，才称明显缓解。否则保留为未解决低标签瓶颈。
5. **Day14/后期权衡**：无论是否通过，完整列出历史方法相对 G-source 及较强 current-only 的 Day14、Day90、Day270 差值与后期恢复保留率。Day14 改善但任一 backbone 的 Day90/270 正恢复保留率低于 0.80，明确记为牺牲后期恢复。
6. **重复且可观测失效**：网站层面，若同一网站的 harmed−corrected 在两个 backbone 都于至少 2/3 dates、每日期至少 2/3 seeds 为正，则记为跨模型重复伤害候选。只有其错误还与预注册的预测时可见 margin/disagreement/prototype-alignment 量呈重复分离，才称存在值得另立实验研究的可观测失效；本轮不据此设计新机制。

## 预算与停止条件

新增 backbone 训练/微调、checkpoint 写入均为 0。不得加入 adapter、attention、memory network、额外表示层、特殊 loss、backbone update、无标签 TTA、外部数据、DNNF/TFAN 或多日期连续适应。只运行一次 source-side 小候选选择和固定 36 个正式配置；不追加 seed、日期、alpha/lambda、solver 或迭代。完成预定产物、独立完整性验证与 `RESULTS.md` 后停止，不自动进入算法设计；研究去留由 Host 决定。
