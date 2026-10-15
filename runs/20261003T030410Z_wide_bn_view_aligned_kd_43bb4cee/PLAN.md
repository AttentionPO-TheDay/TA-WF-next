# 20261003T030410Z_wide_bn_view_aligned_kd_43bb4cee

## 问题

上一轮 raw-logit 集成蒸馏只提升 0.261pp，可能因为教师使用 clean 输入而学生使用 span-mask 输入。本轮固定温度和较低权重，直接比较输入视图对齐及教师分歧门控。

## 条件

- `ce_only`：wide+BN 学生，source-only span-mask，纯交叉熵；历史基线复用用于对照。
- `matched_kd`：学生和三个冻结教师对同一 source batch 使用同一 span-mask；教师在 `T=2` 下的概率平均作为目标。
- `clean_kd`：学生和教师均使用 clean source 输入；教师目标同样是 `T=2` 概率平均。
- `gated_matched_kd`：matched-view；仅当平均教师置信度至少 0.8 且三个教师预测不全相同时计算蒸馏损失。

所有新条件各运行 seed `21729, 23407, 22026`，从头初始化；学生结构、source 行顺序、索引流、训练步数和 valid 选模机会一致。`T=2`、蒸馏权重 `lambda=0.25` 固定，不扫描温度或权重。

## 数据与权限

TemporalDrift source 15300 行、valid 510 行、102 类；沿用上轮冻结 prepared 输入。source 标签用于 CE 和训练期蒸馏；valid 标签只用于预注册 checkpoint 选择与指标。未来日期、WTT-Time、AWF、TTA 和 reference/query 适应关闭。

## 教师与目标

教师是 `20261002T053706Z_progressive_capacity_bn_mixup_1ad233da` 的三个 wide+BN 最佳 checkpoint，参数和 BN 处于 eval/frozen。对每个 batch 计算 `softmax(teacher_logits/T)` 后求三教师算术平均；学生损失为 `CE + lambda*T^2*KL(teacher_prob || student_prob)`，其中 student_prob 使用 `log_softmax(student_logits/T)`。clean 条件复用 clean source logits 仅作为输入对齐的固定目标核验；matched 条件在线计算同 mask 教师输出。

## 训练与判定

wide progressive generator `64/128/256` + BatchNorm，Transformer 2 层、4 heads、d_model 128、FF 256、dropout 0.1；batch 64、12800 steps、AdamW `.001/.0001`，6401/9601 步降学习率。每 seed 在 20 个固定步点评估，valid accuracy 最大且并列取最早 checkpoint；同时报告 accuracy 和 Macro-F1。

候选相对 `ce_only` 需平均 accuracy 至少 +1pp、3/3 seed 正且平均 Macro-F1 不下降。clean_kd 与 matched_kd 的主要比较用于定位视图错配；gated_matched_kd 只有在门控覆盖率被记录后才解释为选择性蒸馏，不使用 valid 标签决定门控。

## 资源与停止

GPU0，最多 3 路并行，单任务 7200 秒、批次 43200 秒；FP32、确定性、无 AMP/TF32。非有限值、教师/学生输入不一致、BN 评估更新、checkpoint 重载错误或预算超限立即停止并保留产物。source/valid 已参与多轮开发，结果不代表独立确认或抗时间漂移。
