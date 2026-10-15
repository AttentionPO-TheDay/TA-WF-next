# 20261004T044803Z_temporal_order_mask_3f751089

## 问题

全局时间原点抖动和局部时间伸缩都未达到稳定收益。本轮测试针对局部时间顺序本身的掩码 `temporal_order_mask`：对 source batch 的连续时间区间，将区间内的 TAM 时间 bin 替换为该区间的通道均值。这样保留两路方向的总计数和区间平均强度，同时隐藏区间内的细粒度时间先后关系。

## 条件

- `ce_control`：沿用既有 90-bin span mask，不增加顺序掩码。
- `temporal_order_mask`：先使用相同的 90-bin span mask；随后以 0.75 概率，在 120 个时间 bin 中随机选择连续 24-bin 区间，将该区间的 24×15 原始位置全部替换为每个方向通道在该区间的均值。双通道使用同一时间区间；区间总计数保持不变，未选中的样本不变。
- 两条件共享模型、初始化、source index stream、span-mask stream、优化器、步数、验证选模规则和三 seed；推理与 checkpoint 评价不应用顺序掩码。

## 数据与判定

TemporalDrift source 15300 / valid 510 / 102 类。source 标签只用于训练；valid 标签只用于预声明 checkpoint 选择和指标。未来日期、WTT-Time、AWF、TTA 均关闭。主比较为 `temporal_order_mask - ce_control`，要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降；否则 FAIL。

## 预算

GPU2，最多两路并行；每任务 7200 秒，总计 43200 秒；FP32、无 AMP/TF32、确定性。复用同一 TemporalDrift source/valid prepared input 的只读副本，不重新划分数据。
