# 20261004T063601Z_temporal_order_components_e212f7ce

## 问题

上一轮时间顺序掩码三 seed 全部提升，但平均 accuracy +0.784pp 未达 +1pp 门槛。本轮在固定区间、概率和训练预算下拆分两个内部组件：掩码强度（硬/软）与替换模板（标量均值/保留 15-position 细粒度形状）。

## 条件

- `ce_control`：既有 90-bin span mask，不增加顺序掩码。
- `hard_scalar`：p=0.75，连续 24 个时间 bin 全部替换为该区间每个方向通道的一个标量均值；对应上一轮方法。
- `soft_scalar`：同一标量均值模板，alpha=0.5，与原始区间按 1:1 混合。
- `hard_shape`：p=0.75，连续 24 个时间 bin 替换为该区间按每个 bin 内 15 个原始位置分别求得的均值模板；alpha=1，保留 bin 内细粒度形状但隐藏 bin 间顺序。
- `soft_shape`：同一细粒度模板，alpha=0.5，与原始区间按 1:1 混合。

所有候选使用相同 24-bin 区间、p=0.75、双通道同步和总计数保持；只在 source 训练使用，valid 与 checkpoint 推理使用原始输入。五条件共享初始化、index stream、span-mask stream、模型、优化器、步数、选模规则和三 seed。

## 数据与判定

TemporalDrift source 15300 / valid 510 / 102 类。source 标签仅用于训练；valid 标签仅用于预声明 checkpoint 选择和指标。未来日期、WTT-Time、AWF、TTA 关闭。每个候选相对 `ce_control` 要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降；否则 FAIL。组件排序以预注册主指标 accuracy 平均提升为准，不根据单个 seed 选择。

## 预算

GPU0，单路顺序执行（无并行）；15 项训练（5 条件×3 seed），每任务 7200 秒，总预算 120000 秒；FP32、无 AMP/TF32、确定性。
