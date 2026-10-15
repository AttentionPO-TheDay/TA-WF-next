# 20261003T122529Z_supervised_contrastive_6dc5c1f8

## 问题

测试监督对比损失是否能在分类 CE 之外约束同一网站的特征聚合，从而提高 valid 泛化。

## 条件

- `ce_control`：与候选相同的模型和投影头，但只优化 CE；投影头用于配对初始化和审计，不参与控制损失。
- `supcon`：`CE + 0.1*SupCon`，temperature=0.1；正样本是同一 source 类别在当前 batch 内的其他样本，self pair 排除，没有正样本的 anchor 从对比损失平均中排除。投影头只在训练期使用，推理只保留分类器。
- 两条件同 seed、初始化、source index/mask stream、AdamW、12800 steps 和选模规则；三 seed 21729/23407/22026。

## 数据与判定

TemporalDrift source15300/valid510/102类；source 标签用于 CE 及获准 SupCon，valid 只用于预声明选模和评价。主比较为 `supcon - ce_control`，要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降。未来、WTT-Time、AWF、TTA 关闭。

## 预算

GPU0，最多两路并行；单任务7200秒，总计43200秒；FP32、无 AMP/TF32、确定性。
