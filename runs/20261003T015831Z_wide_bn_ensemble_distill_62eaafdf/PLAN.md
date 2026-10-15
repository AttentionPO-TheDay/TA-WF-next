# 20261003T015831Z_wide_bn_ensemble_distill_62eaafdf

## 问题

固定三个已经完成并审计的 wide+BN Transformer 作为教师，把它们的 source logits 平均后做软目标，能否提高一个从头训练的 wide+BN 单模型？`shuffled_kd` 保留相同教师输出边际分布但打乱样本对应关系，用于区分样本知识迁移和普通软目标正则化。

## 条件与对照

- `ce_only`：wide+BN 学生，只有 source 交叉熵。
- `ensemble_kd`：交叉熵 + 固定三教师平均 logits 的温度 2 软目标，权重 0.5。
- `shuffled_kd`：同一教师平均 logits，按每个学生 seed 预注册固定 source 行置换后使用。
- 三条件均使用 seed `21729, 23407, 22026`，从头初始化；模型结构、输入、索引流、训练步数和 valid 选模规则相同。

教师来自 `20261002T053706Z_progressive_capacity_bn_mixup_1ad233da` 的 `wide_bn` 最佳 checkpoint。教师只作为冻结历史输入；不重新训练教师，不把集成结果当作学生结果。

## 数据与权限

TemporalDrift source 15300 行、valid 510 行、102 类；沿用已冻结的 prepared 输入和行顺序。source 标签用于 CE、蒸馏和训练指标；valid 标签只用于预先固定的 20 次 checkpoint 选择和指标。未来日期、WTT-Time、AWF、TTA、reference/query 适应均关闭。

## 训练与选模

渐进式 1D CNN 生成器通道 `64/128/256`，每阶段 BatchNorm，Transformer 保持 2 层、4 heads、d_model 128、FF 256、dropout 0.1。batch 64，12800 optimizer steps，AdamW `lr=0.001`、`weight_decay=0.0001`，在 6401/9601 步降到 0.0003/0.0001；source-only span mask p=0.5、连续 90 bins、同方向。每个 seed 在 20 个预注册步点评估，按 valid accuracy 最大值、并列取最早严格改进 checkpoint；同时报告 accuracy 和 Macro-F1。

蒸馏损失为 `(1-w)*CE + w*T^2*KL(log_softmax(student/T), softmax(teacher/T))`，其中 `T=2`、`w=0.5`。教师 soft targets 只由 source 行产生；valid 不产生蒸馏梯度。

## 判定与停止

候选相对 `ce_only` 需平均 accuracy 至少 +1pp、三个 seed 均正且平均 Macro-F1 不下降，才列为开发候选；90% 单独报告。若 CUDA 不可用，不把 CPU 结果伪装成原定 GPU 批次：先运行接口/梯度/教师对齐预检，并记录资源偏差；任何非有限值、输入/标签错位、checkpoint 重载不一致或越过预算立即停止并保留产物。

## 记录

本 run 只复用显式复制的本项目上一轮最小代码、冻结 prepared 输入和历史教师 logits；所有配置、教师文件 hash、索引流、预测、checkpoint 与资源状态归属本目录。该 valid 已参与多轮开发，结果是开发证据，不是独立确认或抗时间漂移结论。
