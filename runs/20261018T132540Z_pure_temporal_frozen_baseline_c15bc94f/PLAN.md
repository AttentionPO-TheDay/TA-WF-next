# 纯净源域模型的 TemporalDrift 冻结退化基线

本 run 是已登记 Temporal 核心矩阵的第一阶段，只计算新三 seed 源域 checkpoint 在 Day14/90/270 上的 `frozen` 结果。五条件核心矩阵仍须独立核对四种适应实现及预算后执行；本 run 的数字可作其配对基线，但不得把旧权重的同名结果冒充新重复。

固定三个 checkpoint：`pure_source_wide_bn_{42,100,2026}.pth`，只按原官方 valid 选模，不用目标日期选模型。目标输入复用项目内已审计的前 5000 有符号相对时间戳及重建 TAM。每个日期复用旧开发 run 的 1024 条无标签 reference 和独立 query 索引，并核对 reference/query 与本次纯净源域训练输入精确隔离。旧索引仅用于固定开发 split，不复用旧 checkpoint 或预测。仅对 query 做冻结前向；query 标签仅用于预测固定后的 Accuracy、固定 102 类 Macro-F1。记录逐样本 logits、预测、熵及 logits L2 汇总。

资源上限：CPU 两线程、9 个 seed×日期单元、总墙钟 10,800 秒、输出 1 GiB。原始数据只读；日志、配置、预测和结果只写本 run。任一 checkpoint/split/输入散列不符、非有限输出或预算越界即停止并保留证据。所有日期都是已观察的 TemporalDrift 开发数据；不存在独立确认宣称。
