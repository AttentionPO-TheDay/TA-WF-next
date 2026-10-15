# 20261014T162930Z_tent_shot_span_f2b7665d

问题：Scale-Lock 冻结 `final_norm`/`classifier`、固定 BN 源域统计量并加入 5% 局部时间 Span-Invariance，能否降低 TENT 的 logit 膨胀和重复 support 过拟合？

## 数据与权限

TemporalDrift Day14/Day90/Day270；沿用已审计 support/query 划分和 3 个 seed（21729、23407、22026），每个日期 1024 条无标签 support，query 仅用于最终评分。source/valid 只做 split 身份核验，不进入适应；不使用 support/query 标签做梯度、选模或阈值调整；WTT-Time/AWF 本轮关闭。

## 方法

每步从 support batch（128）计算无遮挡视图和随机 90-bin（1800 的 5%）局部掩码视图。仅保留 clean entropy 在 0.20–0.60 nats 且两视图 argmax 一致的样本；损失为 clean entropy 与 masked-view 对 clean stop-gradient 概率的交叉熵各 0.5，再乘 800 步余弦衰减。固定 terminal step 800，不早停。

Scale-Lock 配置冻结 `final_norm` 和 `classifier`，BN 使用源域 running statistics（BN eval），只开放 6 个 BN1d 的 affine、`time_projection`、`time_positions` 与 Transformer head 的 4 个 LayerNorm affine。Adam lr=0.00025，betas=(0.9,0.999)，无 weight decay。

## 对照与裁定

复用已完成 TENT、固定统计量、熵门控、全参协同门控和 no_pseudo_fixed 的逐 seed/date 历史结果；新配方 9 个任务。主筛查为平均 Macro-F1 >72% 且相对 tent_bn 配对增益至少 +1.5pp；同时报告 NAR、纠错率、保留率、一致率和 logit 塌缩诊断。结果属于 TemporalDrift 已观察开发证据。
