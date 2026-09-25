# 20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37

问题：排除official-valid重叠并source内canonical去重后，四视角在TemporalDrift五日期的效用是否跨三个训练seed重复

状态：frozen revision 2，数据访问前冻结。

v1在训练前发现固定source与official valid有2条方向内容重叠并按预定条件STOP；未产生性能反馈。本版仅修正source候选隔离规则，其余问题、模型、训练、评价、预算和判读门槛保持v1。

source规则：先计算完整official valid所有行的前5000方向SHA-256；遍历source原行序，排除命中valid的行，对其余相同方向哈希只保留最小行号；再按sampling seed1729逐类随机取20条。保存全部排除行号和数量。valid仍按每类5条，day14/30/90/150/270按每类最多20条。选中source/valid/五日期之间方向哈希交叉必须为0。不得把去重后的剩余source称为全新数据。

权限：source监督，official valid只选epoch；五日期标签仅用于冻结分层抽样与预测封存后评分，不进梯度、标准化、融合或选模。TemporalDrift为已观察开发数据；WTT/AWF和其他Proteus条件不读。

输入：前5000观察；packet方向；50/250固定window比例与转向率；exact direction/log1p count/左右边界；coarse direction/log2 bin/邻居bin差及有效位/左右边界；run width512和显式mask。source统计标准化。

模型：packet两层stride卷积；window 240-128-102 MLP；run两层k5/64卷积、逐层mask、mean/max及128隐藏head。架构和参数量不同，只作效用筛查。

每视角训练seeds1729/3407/2026，共12次从头训练；15epochs/batch128/AdamW lr0.001 wd0.0001。source valid macro-F1最大选epoch、平局取早。CPU4线程、GPU隐藏、timeout1200秒。所有日期/seed/视角完整报告；预测先封存再评分。

主分析为每日期每视角accuracy/macro-F1三seed均值、样本标准差及相对valid下降。若生成视角在五未来日期至少四日的三seed均值accuracy高于packet，标为后续候选正向；门槛不是显著性或确认。不能混入Network/Behavior结果，不能声称等容量、抗漂移机制或selector有效。

实现复制v1训练/核验，仅增加预注册的source隔离修正；不加载旧checkpoint。保存清单、排除审计、统计、模型、历史、预测和哈希。结构/非有限/类别/隔离/完整性错误或超时则STOP。冻结后不根据日期结果续训或调参。
