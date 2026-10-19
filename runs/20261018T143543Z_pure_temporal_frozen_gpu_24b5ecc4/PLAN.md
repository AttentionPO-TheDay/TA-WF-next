# 纯净源域 TemporalDrift 冻结基线完整重跑

前次 `20261018T132540Z_pure_temporal_frozen_baseline_c15bc94f` 在 2/9 单元后进程退出，无错误记录。其产物原样保留。本次用同一三份纯净源域权重、同一 Day14/90/270 输入与旧开发 reference/query 清单，完整重跑九单元；只将计算设备改为 GPU0，绝不复用旧结果。两份已保存 CPU 预测仅在事后核对设备一致性，不参与选模。

输入为前5000有符号相对时间戳与相同规则重建的 TAM；三seed 42/100/2026 的源域权重由已观察 official valid 选出。1024 条 reference 仅用于划分验证；query 与源域 source/valid 及 reference 逐输入精确隔离。模型完全冻结，query 标签只用于终端 Accuracy、固定102类 Macro-F1；记录逐样本预测、logits、熵和logits L2。TemporalDrift日期均为已观察开发数据，不称独立确认。

预算：GPU0 单进程、2 CPU线程、最大3600秒、九单元、输出≤1GiB。代码和输入散列在第一次目标前向前冻结。任何数据划分或权重不符、非有限输出、超时立即失败并保留产物。目标对照仍为五条件核心矩阵；本 run 只建立其 `frozen` 配对基线，其他条件须单独完成实现核对并冻结。
