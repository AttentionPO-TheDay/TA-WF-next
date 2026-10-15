# 20261004T011930Z_local_time_stretch_aug_302fcd01

## 问题

上一轮全局时间原点平移仅带来 +0.196pp 平均 accuracy，未通过一致性门槛。本轮测试更贴近局部时间漂移的输入增强 `local_time_stretch`：对 source batch 中的 TAM 时间轴局部区间做同步压缩或展开，并按每个方向通道的增强前总计数重标定，保持总计数不变。该增强只用于 source 训练，valid 保持原始输入。

## 条件

- `ce_control`：沿用既有 90-bin span mask，不增加局部伸缩。
- `local_time_stretch`：先使用同样的 span mask；随后以 0.75 概率，在 120 个时间 bin 中随机选择一个连续 24-bin 区间，对两路 TAM 使用同一局部坐标映射，随机因子为 0.8 或 1.25。映射使用线性插值，边界坐标裁剪；每个方向通道按增强前后全轴总计数比例重标定，避免增强改变方向总量。未选中的样本保持不变。
- 两条件共享模型、初始化、source index stream、span-mask stream、优化器、步数、验证选模规则和三 seed；推理与 checkpoint 评价不应用伸缩。

## 数据与判定

TemporalDrift source 15300 / valid 510 / 102 类。source 标签只用于训练；valid 标签只用于预声明 checkpoint 选择和指标。未来日期、WTT-Time、AWF、TTA 均关闭。主比较为 `local_time_stretch - ce_control`，要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降；否则 FAIL。

## 预算

GPU2，最多两路并行；每任务 7200 秒，总计 43200 秒；FP32、无 AMP/TF32、确定性。复用上一轮同一 TemporalDrift source/valid prepared input 的只读副本，不重新划分数据。
