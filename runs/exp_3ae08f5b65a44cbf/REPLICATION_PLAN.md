# Backbone-seed replication pre-registration

本文件在任何新增 backbone 训练开始前冻结。TemporalDrift 是已观察的机制开发数据；本实验只重复既有强常规 baseline，不开发选择器、更新机制或模型结构。结果出来后不得修改本计划；实现错误只允许保留证据后修复并按完全相同配置恢复相同 seed。

## 来源与冻结证据

- 训练协议与原正式 checkpoint：`runs/exp_9121b664a1854097`。原正式 training seed 为 `6238`；DF epoch 29 checkpoint SHA-256 `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`，VarCNNDirection epoch 23 checkpoint SHA-256 `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83`。原 checkpoint 只读且不重训。
- 正式 v3 source split：`runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`，SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`。角色固定为 supervised-train 16,309、reference 204、source-holdout 2,040；official `valid.npz` 全体只用于 source checkpoint 选择。
- current support donor：`runs/exp_b471517a3e6f41e7/artifacts/support_manifests.json`，SHA-256 `cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca`；隔离审计 SHA-256 `6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6`。直接复用 Day14/90/270 × 3/10-shot × support seeds `1729/6238/20260916`，不重新抽样或重建审计。
- selection 规范：`runs/exp_376fca9354214097/SUPPORT_SELECTION_PLAN.md`，SHA-256 `89c3059fefc78fe47151e8a25e8107f64171542597d87af335539e401e3791dc`；正式 common-query manifest SHA-256 `161ebc6e71885258808a865c97c346abdc77fba3b823e595ba510e8159e531ed`。原 seed 的 36 个正式结果优先只读复用，并逐文件校验 `integrity_check.json` 所列 SHA-256。
- 数据文件 SHA-256 固定为 train `2994271abc43bc3e3513924367da475e0a724b2cdfce9212968f6dc05763c882`、valid `e0f3ea5b170ecd9476c70dbc24c712266f014727b43fb8a377a546eed5c7ab62`、Day14 `eaa52ab83aea590ebc336cac1e8c2c6df8ea37ecc5ea26d5686e951b18460333`、Day90 `eaa25c23657f22509ab3ac9018a63a12736fab007db45cfa2ea05cbb6c0b25a6`、Day270 `35e965b9eed0239ca9b4959228defb85626235c88ef77b9b83c842a32bab9488`。

入口在训练与评价前逐项复核上述文件以及旧 integrity、history/checkpoint metadata、support/common-query row 与 content-hash 关系。任一 donor、split、manifest、规则或旧正式 evaluation hash 不一致，立即停止且不自行重建。

## 训练 seeds、配置与固定预算

项目没有规定额外 backbone training seed 列表，故在训练前一次性预注册两个常规整数 seed：`1013` 与 `2024`。它们均不同于原正式 seed `6238`，不得根据训练或 TemporalDrift query 结果替换。每个 architecture 最终汇总 training seeds `[6238,1013,2024]`。

新增训练严格为 4 次：DF×1013、DF×2024、VarCNNDirection×1013、VarCNNDirection×2024。每个 seed 只允许一次逻辑训练；异常退出可用完全相同配置恢复该 seed，不得补训第三个 seed、延长 epoch、调整超参或择优丢弃。

- 模型：仓库 `src/ta_wf_next/models/df.py::DF(102)` 与 `src/ta_wf_next/models/varcnn.py::VarCNNDirection(102)`；后者是方向单分支派生，不称为完整双分支 VarCNN。
- 输入：TemporalDrift 有符号时间戳前 5000 位逐元素取 `sign`，正为 +1、负为 -1、0 为 0；形状 `[B,1,5000]`，不解释成包大小。
- 数据：只用 v3 `supervised_train` 训练；不得重划分。official `valid.npz` 全体只用于 source checkpoint 选择。reference、source-holdout 与未来日期均不进入训练或 source 选模。
- 初始化/随机性：对每一预注册 training seed 调用现有 `seed_everything(seed)`，同时以同一 seed 初始化训练 DataLoader shuffle generator；cuDNN deterministic=true、benchmark=false。这是相对原正式协议唯一变化的随机 seed 实例化。
- 优化：AdamW，learning rate `0.001`，weight decay `0.0001`，batch size `64`，CrossEntropyLoss，最多且固定跑满 `30` epochs；validation batch 128；不早停。
- checkpoint：逐 epoch 在 official source validation 上计算 macro-F1；取 macro-F1 最大 epoch，完全并列取最早 epoch。保存 best checkpoint，但仍完成 30 epochs。不得以未来结果选 checkpoint。

每个新增 checkpoint 单独记录 architecture、training seed、30-epoch history、best epoch、source-validation accuracy/macro-F1、checkpoint 相对路径、SHA-256、模型定义文件 SHA-256，以及包含 seed、split/data hashes、输入、optimizer、lr、weight decay、batch、epoch 和选择规则的训练配置 SHA-256。

## 冻结表示、G-source 与候选

每个 checkpoint 固定 `eval()`，dropout 关闭、BN 使用自身 checkpoint 统计。最终 global embedding 是 `forward_features`/`forward` 第二返回值、原 `mlp` 输入的 512 维输出。全部 G/prototype 方法使用逐行 L2 归一化 embedding `z`，零向量保持零。

每个新增 checkpoint 独立重建自己的 source-side 对象，但规则不变：source prototype 只由 v3 supervised-train 的归一化 embedding 按类均值后再归一化；G-source 只在 supervised-train 拟合带截距 L2 multinomial logistic regression，候选 `C in {0.1,1,10}`，只按 official source-validation macro-F1 最大选择，完全并列取较小 C。solver=`lbfgs`、max_iter=300、tol=1e-4、class_weight=None、fit_intercept=true、random_state 等于该 backbone training seed。若达到迭代上限仍保留并记录，不扩大迭代或换 solver。

正式 support-internal candidate、数学定义和 solver 完全复用 `exp_376fca9354214097`：

- prototype family：`G_source`、`proto_alpha_0.25`、`.50`、`.75`、`current_prototype`。source/current prototype 都由归一化 embedding 的类均值再归一化；插值后再次归一化；current-only 等价 alpha=1。G-source 是线性 no-update 端点，不以 source prototype alpha=0 替代。
- linear family：`G_source`、`shrink_lambda_1.00`、`.10`、`.01`、`G_current`。Shrinkage 从该 checkpoint 自己的 G-source `(W0,b0)` 初始化，float64 SciPy L-BFGS-B、解析梯度、maxiter=100、maxls=20、ftol=1e-9、gtol=1e-5。G-current 从头仅在相应 support 拟合与 G-source 同形式且继承其 source-selected C 的 logistic head。
- 不增加 alpha/lambda/C、solver、迭代、端点、adapter、损失或表示层。

## support CV、tie-break、重拟合与 common query

每个 architecture×training-seed×date×shot×support-seed 独立执行相同 support-only CV。每类 support row 按 row index 升序，以类内位置 `j mod K` 分 fold：10-shot 使用 K=5（每类 8 train/2 validation），3-shot 使用 K=3（每类 2 train/1 validation）。无额外随机性。

family 内有效候选依次按：(1) fold mean macro-F1 高；(2) 完全相等时 fold mean accuracy 高；(3) 仍相等时 fold macro-F1 样本 SD 低；(4) 保守顺序。Prototype 顺序为 `G_source,.25,.50,.75,current_prototype`；linear 顺序为 `G_source,lambda=1,.10,.01,G_current`。跨家族 `support_selected_baseline` 合并去重候选并用相同前三项，最终顺序固定为 `G_source,alpha=.25,lambda=1,alpha=.50,lambda=.10,alpha=.75,lambda=.01,current_prototype,G_current`。选中后在该配置全部可用 support 上重建/重拟合；G-source 不拟合。

每个 date×support-seed 的 common query 固定为 canonical eligible pool 排除完整 10-shot support。3-shot 与 10-shot 在同一 common query 评分；3-shot 只用嵌套 3-shot 做 CV 和 full-support refit，剩余 7-shot 不训练、不选参，也不返回 query。程序必须逐项核验 donor manifest 与既有 common-query hash。

对一个 architecture×training-seed×date，必须先完成六个 shot×support-seed 配置的全部候选 CV、full-support refit 与 common-query 预测并原子落盘，之后才可读取 query truth 评分。query 标签不得进入候选、fold、C、family、停止、网站筛选或任何预测决策。

## 终点、指标与输出

10-shot 是 replication 主终点。3-shot 只重复既有低标签 selection-noise 诊断；无论结果如何均不得修改方法、增加标签或候选。每个 architecture×training-seed×date×shot×support-seed 至少保存并报告：`G_source`、`current_prototype`、`G_current`、固定 alpha=.25、固定 lambda=.10、support-selected prototype、support-selected linear、跨家族 `support_selected_baseline`；保存所有候选预测和 CV 诊断。

每个方法报告 accuracy、macro precision/recall/F1、102 类指标，以及相对 G-source 的 corrected、harmed、net 计数/比例。按 support seeds 与 training seeds 分层汇总。原 seed 6238 直接复用 `exp_376fca9354214097` 正式结果并校验 artifact hash；不重复计算。

3-shot/10-shot 均报告 selected-oracle 一致率、posthoc query-oracle regret、winner-runner CV margin、selected fold SD、fold-wise winner agreement 和选择频率；oracle 只在预测固定后用于归因，不是方法成绩。

## 两类随机性与方差分解

对 accuracy、macro-F1 及相对 G-source/current-only 的配对效应，至少执行：

1. Backbone training randomness：固定 architecture/date/shot/support-seed/method，跨 `[6238,1013,2024]` 报均值、样本 SD（ddof=1）与范围。
2. Support sampling randomness：固定 architecture/training-seed/date/shot/method，跨 `[1729,6238,20260916]` 报均值、样本 SD 与范围。
3. 对每个 architecture×date×shot×method 的平衡 3×3 表，拟合描述性两因素加性分解 `y_ij=mu+T_i+S_j+e_ij`，报告 training-seed、support-seed 与不可分离的 interaction/residual 平方和、均方及总变异占比；同时报告两因素 cell 表。每格只有一次观察，故 interaction 与残差不能分开，不作显著性检验或过度推广。

support sampling seed 与 backbone training seed 始终作为两个不同字段和重复单位，不相互替代。

## 预注册 10-shot 成功/失败门槛

为避免事后移动边界，沿用 `exp_376fca9354214097` 的 macro-F1 effect-size 定义，并将它逐 training seed 应用。先在每个 architecture×training-seed×date 上对三个 support seeds 取未加权均值。

- Day14 保护：selected 不低于 G-source 超过 1.0 pp，且 selected 至少比 `max(G_current,current_prototype)` 高 2.0 pp。
- Day90/270 recovery：`R=max(G_current,current_prototype)-G_source`，`H=selected-G_source`。要求 R>0 且 H/R≥0.80；若 R≤0，记为该 representation seed 没有可复现的 current-only recovery，不能用负分母制造有利结论。
- 严格 replication success：两个 architecture 的全部三个 training seeds 均通过 Day14，且其 Day90 与 Day270 均通过 recovery，即 6/6 Day14 cells 与 12/12 late-date cells 全通过。任何一个 cell 未过即为严格失败；边界接近也不改门槛。accuracy 同时完整报告但不替代 macro-F1 判定。

由于原 seed 6238 的既有 DF Day14 正式结果已知距旧 2.0 pp 门槛 0.049 pp，本标准不会追溯改写旧结论；该 cell 仍按精确正式值判定。另单独报告两个新增 seed 的通过模式、跨 seed SD/range 和离门槛距离，供 Host 区分“已知边界失败是否普遍”与“训练随机性下的系统失败”，但不另造替代成功标准。

若严格 replication 不成立，按预注册顺序描述性归因：(A) 若 source-valid macro-F1 相对三 seed architecture 均值偏离超过 2.0 pp 或跨 seed range 超过 4.0 pp，标记 source checkpoint/representation quality 明显变化候选；(B) source quality 未明显变化但 selected 相对端点的保护/恢复效应跨 training seed range 超过 5.0 pp、符号改变或过线状态改变，标记 selection benefit 对 representation seed 敏感；(C) 若失败仅限不超过全部主 cell 的 1/6，且每个失败 cell 只有一个 support seed 方向异常或 seed-mean 距门槛不超过 0.25 pp，标记少数 date/support-seed 边界波动；条件可重叠，完整数值优先，不强行唯一归因。不得据此在本实验设计新 adapter 或更新机制。

## 信息权限与停止条件

WTT-Time、AWF 及任何其他封闭最终评价数据在本实验中保持不可访问：不做文件检查、解封、训练、调参或评分。TemporalDrift 仅作开发/机制重复性验证。

新增代码、checkpoint、日志、预测、配置、指标与报告只写入 `runs/exp_3ae08f5b65a44cbf/`；旧 checkpoint 与 artifacts 只读。完成四次新增训练、两个新增 seed 的固定评价、三 training-seed 汇总、独立 integrity verification 与 `RESULTS.md` 后停止。若 replication 成立，只把下一候选问题表述为：每类仅 3 条 current 标签时，能否达到常规 10-shot maintenance 的大部分恢复并保持早期性能；不预设任何机制有效，也不自动启动新算法开发。
