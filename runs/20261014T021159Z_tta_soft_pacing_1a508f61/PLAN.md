# 20261014T021159Z_tta_soft_pacing_1a508f61

验证问题：在 TemporalDrift Day14/90/270 上，固定 Adam 学习率 0.00025 和 800 次候选更新，只给当前无伪标签基线增加 Soft Pacing Wrapper，能否把负适应率降到 3% 以下并保持纠错率？

Wrapper 使用分层源参数锚定（generator/time_projection 权重 0.02，其余参数 0.002）、边缘预测多样性项 0.4、BN momentum 0.05，并屏蔽已审计为零的 classifier 前 128 个输入列。基础损失为 source CE + target entropy + feature MMD，边缘项只由 Wrapper 计算一次。

数据权限：TemporalDrift 已观察开发集；复用既有 support/query 精确划分，不重新随机划分。support 不使用标签，query 只用于最终评分；不使用伪标签、GMM 或 query 梯度。3 seed×3日期共9个新单元，冻结终点为 update 800；历史 frozen/no_pseudo_fixed/dynamic_lr_only 仅作参考。

主指标为 Macro-F1、负适应率和纠错率；同时记录 accuracy、更新时间、反向耗时、显存和 Wrapper 各损失项。筛查标准为负适应率 <3% 且纠错率不低于 no_pseudo_fixed 低1个百分点；不根据 query 结果调参。WTT-Time/AWF 和其他因素关闭。
