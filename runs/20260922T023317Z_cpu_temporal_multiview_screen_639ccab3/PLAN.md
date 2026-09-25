# 20260922T023317Z_cpu_temporal_multiview_screen_639ccab3

问题：packet、window与保序exact/coarse run在真正TemporalDrift日期上的效用和漂移退化是否跨三个训练seed重复

状态：frozen，访问TemporalDrift数组前冻结。

目标：首次在真正的TemporalDrift日期上筛查packet、windows与保序exact/coarse run的轻量学习效用及退化轨迹。TemporalDrift/train按seed1729每类20条作监督，valid每类5条选择epoch；day14/30/90/150/270各每类最多20条只作开发评分。未来标签仅用于固定分层抽样和全部预测封存后的评分，不进入梯度、标准化、融合或选模。外部WTT/AWF与其他Proteus条件不读。

四视角沿用前序实现：packet前5000方向；windows为50/250固定槽的正向比例和转向率；exact为direction/log1p count/左右边界；coarse为direction/log2 bin/邻居bin差及有效位/左右边界。run width512、mask显式；末token的next-bin规则沿用。source逐位置或有效token标准化，未来只应用source统计。

模型沿用已开发CPU架构：packet两层stride卷积+adaptive pool；window 240-128-102 MLP；run两层k5/64通道卷积、逐层mask、masked mean/max与128隐藏分类头。架构/参数量不同，不能将视角差值归因于表示本身或称等容量比较。

每视角训练seeds 1729/3407/2026，共12次训练，从头初始化；15epochs、batch128、AdamW lr0.001 wd0.0001，CPU4线程、GPU隐藏，timeout1200秒。source valid macro-F1最大选epoch，平局取早；所有日期、视角、seed完整报告。源/valid/未来选中方向哈希交叉必须为0，否则停止。

主输出为每日期每视角accuracy/macro-F1的三个seed均值和样本标准差，以及相对各自valid的absolute drop。开发判读：若某生成视角在五个未来日期至少四个日期的三seed均值accuracy高于packet，记为候选正向；否则不称稳定优于packet。该门槛只用于后续开发优先级，不是统计显著性或外部确认。不能把Network/Behavior既有结果混入本批重复。

实现显式复制修改本项目cpu_run_order_ablation训练器，不加载旧checkpoint，仅借用旧venv依赖。预测先封存后评分，保存抽样、源统计、checkpoint、历史和代码哈希。超时、非有限、类别不全、隔离或完整性错误则失败并保留记录。冻结后不根据日期结果续训或改超参；后续变更另立实验。
