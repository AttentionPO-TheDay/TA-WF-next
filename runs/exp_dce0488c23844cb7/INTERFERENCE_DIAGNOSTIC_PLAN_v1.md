# 共享更新干扰存在性诊断：冻结计划 v1

冻结时间：2026-09-19（在本 Job 产生或打开任何评价结果之前）。证据角色：TemporalDrift development evidence。第一关的真网站身份是 oracle diagnostic upper bound，不是部署方法。

## 1. 固定资产与选择依据

- 唯一日期：TemporalDrift Day90，`/mnt/data2/ren/datasets/TemporalDrift/day90.npz`。原因不是本 Job 表现，而是旧 DF Tent 诊断唯一系统使用并明确标记 `selection_day=90` 的开发日期；旧 `df_tent_groups_day90.json` 已固定 lr 0.005、5 步、阈值 0.5。
- 唯一 checkpoint：`/home/rbf/TA-WF/outputs/df_source_tent_seed3407_pilot-20260905-194436/best.pt`，epoch 30，seed 3407，训练仅用 TemporalDrift source，source-valid macro-F1 选模。执行前记录 SHA-256；不得加载其他 checkpoint。
- 唯一输入：原始有符号相对时间戳前 5000 位，float32，shape `[B,1,5000]`，不取 sign、不当作包大小。该输入与旧 checkpoint/config 相配。当前新工作区 sign-only checkpoint 与之不兼容，禁止混用。
- 唯一 TTA：episodic support-only entropy minimization；模型 eval，保留 source BN running statistics，Dropout 关闭；仅更新全部 BatchNorm affine gamma/beta；SGD 等价函数式更新，lr 0.005，5 步；每步只用当前 max-softmax >=0.5 的 support；第五步后 support mean entropy 未上升才接受，否则完整回退 source 参数。query 输入/标签不参与更新或接受判定。
- 依据：旧配置 `df_source_tent_pilot.yaml`、旧实现 `ta_wf/adaptation/tent.py` 和已完成 Day90 group/scope/response artifacts。只显式迁入完成本诊断所需的最小代码并作行为测试，不通过 `sys.path` 依赖旧工程。

## 2. 数据隔离与预算

- 类别 0--101 视为 102 个真实网站身份。仅第一关可用身份构造 oracle 组。
- seed `20260919`；用 `numpy.random.Generator(PCG64(seed))` 对每个网站的 Day90 row indices 独立排列。前 64 条为 adaptation pool，其余为 evaluation pool。已核对每类至少 239 条，故无需按结果删类或缩组。
- 每个 oracle A/B 是单网站组；A-only、B-only、mixed 的总 adaptation budget 都严格为 64。A-only/B-only 各取本网站 64 条；mixed 取 A 排列前 32 条与 B 排列前 32 条，并用固定 mixed seed 打乱顺序。更新步数、筛选、BN 状态和接受规则相同。
- evaluation 使用该网站除前 64 条外的全部独立样本。标签仅在所有 baseline/updated logits 固定后计算指标和转移类型；不进入更新、筛选、接受、配对、阈值或参数选择。
- 每个条件从 checkpoint 重新构造模型与 source 参数；不同 pair、source 或随机重复之间不继承参数、optimizer 或 BN buffer。

## 3. 覆盖性配对与随机对照

- oracle 配对在看结果前固定为 `(0,1),(2,3),...,(100,101)`，共 51 个不重叠 pair，覆盖全部 102 网站。每个 pair 同时报 A→A、A→B、B→B、B→A 及 mixed→A/B；不得结果后重排或只报正例。
- 大小匹配随机对照固定 5 次，seeds 为 `20260920..20260924`。对每个 oracle pair，分别将该 pair 的 128 adaptation rows（每站 64）和两站全部 evaluation rows作 label-blind等大小二分，形成 random A/B；每个 random 组大小与对应 oracle 组相同。A-only/B-only/mixed 仍各用总 adaptation 64。五次均完整报告，不挑最好重复。
- 主统计单位为 directed group（oracle 102 个）；random 分布按 5 个完整重复分别汇总，避免把同一 checkpoint 下样本当独立训练重复。

## 4. 第一关指标和预注册判据

逐 update-source × recipient 保存：accuracy、mean true-class probability、wrong→correct、correct→wrong、wrong→different-wrong、unchanged-correct、样本数、每步/总有效筛选数、每步梯度 L2、最终参数 delta L2、相对 delta L2、support entropy before/after 和是否接受。Macro-F1 仅对相应正式 evaluation predictions 拼接后计算；禁止平均单网站 F1。

令每个 directed oracle group 的 `self` 为本网站 update 对本网站 eval 的 delta，`cross` 为同一 update 对配对网站 eval 的 delta；`mixed` 为 mixed update 对该 recipient 的 delta。第一关只有以下全部成立才 PASS：

1. Self usable：102 个 self 的 mean accuracy delta >= +0.005，mean true-class-probability delta >= +0.002，且至少 60% 的 directed groups accuracy delta > 0。
2. Cross damage：102 个 cross 的 mean accuracy delta <= -0.005，mean true-class-probability delta <= -0.002，且至少 60% 的 directed groups accuracy delta < 0。
3. Mixed cancellation：在 self accuracy delta >0 的 directed groups 中，至少 60% 满足 `mixed_delta <= 0.5 * self_delta`；且全体 mean mixed accuracy delta <= 0.5 * mean self accuracy delta。
4. Oracle exceeds ordinary sampling：interference contrast `mean(self_accuracy_delta - cross_accuracy_delta)` 严格大于 5 个 random repeats contrast 的最大值；oracle self-positive/cross-negative 同向比例也严格大于 5 个 random repeats的最大值。
5. 不是简单幅度解释：所有接受 update 的有效筛选比例均报告；A/B/mixed 三源 median 有效比例最大差 <=0.10，median relative parameter-delta 最大/最小比 <=2；在 102 directed oracle updates 上，`abs(Spearman(contrast, effective_fraction)) <0.5` 且 `abs(Spearman(contrast, relative_parameter_delta)) <0.5`。任何一项不可计算或失败均不支持继续。

若任一项失败，唯一第一关裁决为 `STOP_NO_USABLE_INTERFERENCE`，立即停止第二关。不得改 lr、步数、阈值、预算、配对、日期、checkpoint 或输入来寻找正例。

## 5. 第二关（仅第一关 PASS）

只在 PASS 后计算：原始预测分布 Jensen-Shannon 相似度、frozen 512-d embedding centroid cosine/distance、support entropy-loss BN-affine gradient cosine、A update 对 B 原始预测分布的 mean JS change、以及 mean max-softmax confidence baseline。评价标签只在信号全部固定后定义实际 cross accuracy/true-probability gain/damage。

不拟合复杂 selector，不调阈值。每个信号报告 Spearman 方向、leave-one-pair-out 可审计 median split balanced accuracy/AUROC（若两类均存在）和相对 confidence baseline。稳定预测要求预先定义为：预期方向的 Spearman |rho|>=0.30、permutation p<0.05、median-split balanced accuracy>=0.60，并且 51-pair leave-one-pair-out方向一致率>=0.60；至少一个非置信度信号满足且 balanced accuracy 比 confidence 高 >=0.05，才裁决 `SUPPORTS_SHARED_SCOPE_METHOD_FOLLOWUP`，否则 `INTERFERENCE_EXISTS_BUT_NOT_DEPLOYABLE`。熵下降不作为分类改善。

本 Job 不含 Burst、selector 设计、新训练、外部测试或后续方法启动。

## 6. 资源和停止

- 新 backbone 训练 0；新 checkpoint 0；超参数搜索 0。仅现有 checkpoint 的临时函数式 TTA 副本。
- 优先单 GPU；允许 CPU 单元/一致性测试。预计 51 oracle pairs + 5×51 random controls；如资源或完整性检查失败，记录并停止，不缩减覆盖范围冒充完成。
- 所有新文件仅写入 `runs/exp_dce0488c23844cb7/`。
