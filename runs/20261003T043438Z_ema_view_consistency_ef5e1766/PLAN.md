# 20261003T043438Z_ema_view_consistency_ef5e1766

## 问题

固定同结构教师的输出蒸馏在前两轮只有约 +0.261pp 且不稳定。本轮测试不使用固定教师—学生蒸馏，而使用学生参数的在线指数移动平均（EMA）副本作为一致性参考：学生从强 span-mask 输入预测，EMA 副本在同视图或 clean 视图上预测，约束两者输出一致。

## 条件

- `ce_only`：wide+BN 学生，source-only span-mask，纯交叉熵。
- `ema_same_view`：学生和在线 EMA 副本接收完全相同的 span-mask batch；加入 EMA 概率一致性损失。
- `ema_cross_view`：学生接收 span-mask，EMA 副本接收同一 source batch 的 clean 输入；加入跨视图概率一致性损失。

EMA 副本不是预训练教师，不加载历史 checkpoint，不参与反向传播；每步 optimizer 更新学生后，以 `decay=0.999` 更新 EMA 参数。EMA 模型处于 eval，BN buffer 每步从学生同步，避免独立统计漂移。

## 固定机制与损失

`T=2`；最大一致性权重 `lambda=0.25`，从第 1 步线性 ramp 到第 3200 步，之后保持 0.25。损失为 `CE + lambda(step)*T^2*KL(EMA probability || student probability)`。EMA 目标始终 detach；不使用 valid 标签决定门控或梯度。

## 数据与权限

TemporalDrift source 15300 行、valid 510 行、102 类；source 标签用于 CE，valid 标签只用于预注册 checkpoint 选择和指标。未来日期、WTT-Time、AWF、TTA 和 reference/query 适应关闭。

## 训练与判定

三条件各运行 seed `21729, 23407, 22026`，wide+BN 结构、batch 64、12800 steps、AdamW `.001/.0001`、固定 source index stream 和 span-mask RNG。valid 在 20 个固定步点评估，按最高 valid accuracy、并列取最早 checkpoint。候选需相对 CE 平均 accuracy 至少 +1pp、3/3 seed 提升且平均 Macro-F1 不下降，才进入后续。

## 资源与停止

GPU0，最多 3 路并行，单任务 7200 秒、批次 43200 秒；FP32、确定性、无 AMP/TF32。非有限值、EMA/学生输入形状错误、EMA 梯度泄漏、checkpoint 重载错误或预算超限立即停止并保留产物。source/valid 已用于多轮开发，结果不代表独立确认或抗时间漂移。
