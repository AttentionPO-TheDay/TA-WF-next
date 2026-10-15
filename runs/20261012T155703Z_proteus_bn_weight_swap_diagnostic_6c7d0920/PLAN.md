# Proteus 0.25 倍 checkpoint 权重/BN 统计交换诊断

问题：低强度 Proteus 的收益主要来自可学习参数变化，还是 BatchNorm running statistics 的变化？

复用已完成的 `strength_025` checkpoint，不重新训练、不重新选择适应步骤。对每个 seed×Day14/90/270 构造四个推理状态：frozen（原始权重和 BN 统计）、weights_only（适应后参数，原始 BN 统计）、bn_stats_only（原始参数，适应后 BN running statistics）、full_strength025（完整适应 checkpoint）。BN affine 参数属于参数状态，只有 running_mean、running_var、num_batches_tracked 属于统计状态。

query 标签只在固定推理后计算 Macro-F1、accuracy、负适应率和纠错率；本诊断不访问 support，不产生梯度，不改变 checkpoint。四种状态不能简单相加，结果用于归因诊断。TemporalDrift 已观察开发数据，外部数据关闭。
