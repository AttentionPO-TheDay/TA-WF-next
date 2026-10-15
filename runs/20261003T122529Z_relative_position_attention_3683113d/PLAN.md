# 20261003T122529Z_relative_position_attention_3683113d

## 问题

当前 wide+BN Transformer 使用可学习绝对 token 位置。测试固定相对距离注意力偏置是否能让模型利用 token 间局部/远距离关系，提高 valid 泛化。

## 条件

- `ce_control`：原 wide+BN Transformer，保留同一个非训练 relative-bias buffer 但不把它传入注意力。
- `relative_position`：两层 Transformer 都加入 `-0.10*log1p(abs(i-j))` 的 220×220 固定相对距离偏置；偏置不训练、不按 valid 调整。
- 两条件同 seed、初始化、source index/mask stream、AdamW、12800 steps 和选模规则；三 seed 21729/23407/22026。

## 数据与判定

TemporalDrift source15300/valid510/102类；source 标签仅用于 CE，valid 仅用于预声明选模和评价。主比较为 `relative_position - ce_control`，要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降。未来、WTT-Time、AWF、TTA 关闭。

## 预算

GPU0，最多两路并行；单任务7200秒，总计43200秒；FP32、无 AMP/TF32、确定性。
