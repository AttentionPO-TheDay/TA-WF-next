# Proteus 低学习率损失归因实验 v1

问题：完整 Proteus 在 0.001 学习率下收益不稳定；0.00025 倍学习率产生更高 F1。为判断低强度有效配方的收益来自哪些损失，重新执行完整损失组件消融。

## 条件与控制

六个条件：full_control、no_pseudo、no_mmd、no_entropy_diversity、no_source_ce、source_ce_only。所有条件保持三路 source/target/pseudo 前向、GMM 刷新、BN/dropout 行为、source/target 抽样流和 800 次更新完全一致，仅改变损失权重。Adam 学习率固定为 0.00025；每个任务从同一个 wide+BN source checkpoint 开始。

no_pseudo 只去除伪标签 CE 梯度，仍保留 GMM 与目标前向；source_ce_only 仍保留目标与伪标签前向及 BN 更新，因此不等同目标域完全关闭。

## 数据权限

仅使用 TemporalDrift Day14/90/270 已观察开发数据。support 只含输入与行号，support 标签不进入适应；source 标签仅用于 source CE；query 只在适应结束后推理并评分。外部未来、WTT-Time、AWF 及其他因素关闭。

## 裁决

每个消融相对本轮低学习率 full_control 报告 Macro-F1、accuracy、负适应率、纠错率及 9 个配对单元。筛查信号为 F1 平均提高至少 0.5pp、负适应率下降至少 1pp、且至少 6/9 单元 F1 提高。该实验用于条件性归因，不把单次组件差值解释为唯一因果贡献。

## 预算与审计

GPU1 单进程，54 项新任务，每项 100 epoch/800 更新；保存 loss、GMM、streams、checkpoint、query logits，独立审计冻结 hash、指标、stream、更新和重载预测。

## 修订

来源于 `runs/20261010T072743Z_proteus_loss_ablation_a37b4fe6` 的最小实现；仅学习率由 0.001 改为预注册有效强度 0.00025，其他机制不变。
