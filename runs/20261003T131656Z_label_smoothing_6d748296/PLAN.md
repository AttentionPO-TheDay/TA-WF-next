# 20261003T131656Z_label_smoothing_6d748296

## 问题

当前 wide+BN source accuracy 接近完全拟合而 valid 仍低于训练集。测试 source CE 使用固定 `epsilon=0.05` 标签平滑能否改善 valid 泛化。

## 条件

- `ce_control`：标准硬标签 CE。
- `label_smoothing`：`F.cross_entropy(..., label_smoothing=0.05)`；valid 仍用硬标签 accuracy/Macro-F1 评价。
- 两条件完全相同的模型、初始化、source index/mask stream、AdamW、12800 steps 和选模规则；三 seed 21729/23407/22026。

## 数据与判定

TemporalDrift source15300/valid510/102类；source 标签只用于训练，valid 只用于预声明选模和评价。主比较为 `label_smoothing - ce_control`，要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降。未来、WTT-Time、AWF、TTA 关闭。

## 预算

GPU2，最多两路并行；单任务7200秒，总计43200秒；FP32、无 AMP/TF32、确定性。
