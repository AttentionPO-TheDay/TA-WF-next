# 20260923T074303Z_gpu_tail_neutral_convergence_6e4e1ab5

问题：固定抽样与共享标准化下，tail-neutral相对full的valid与未来开发日期收益在45轮训练后是否保持？

状态：frozen，训练前登记。

输入：从 `configs/datasets.json` 定位 TemporalDrift，沿用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 的 source/valid/Day14/30/90/150/270 行号。生成器沿用本项目 `traffic_views.py`，5000 个方向预算、50/250 宽窗口。full 和 tail-neutral 的唯一输入差异为 partial window 两个比例是否改为 (0.5, 0)；共用 full source 均值/标准差。

训练：同一个 240→128→102 MLP、AdamW 1e-3、weight decay 1e-4、batch 128；从头初始化，三 seed 1729/3407/2026，每条件 45 epochs。GPU 0，仅一块卡；900 秒上限。禁止加载旧 checkpoint。source 标签只用于拟合，valid 标签用于每个条件和 seed 的 macro-F1 最高且并列最早 epoch 选择；未来日期标签只用于选模后开发评分，不适配、不反向传播。

主比较：tail-neutral 减 full 的 valid 与各日期 macro-F1（百分点），以及 valid→Day270 绝对下降的差。候选门槛：valid 平均差>0 且至少 2/3 seed 为正；五个未来日期平均差>0 且至少 4/5 日期均值为正。抗漂移门槛另要求 valid→Day270 下降缩小，且 Day270 绝对 F1 不下降。报告所有 seed、所有日期，不以最优 seed 代替平均。

停止：配置非 frozen、CUDA 不可用、已存在最终预测、输入非有限、抽样数量或类别不符时训练前终止。数据外部 WTT-Time/AWF 保持关闭。该实验是对已观察 TemporalDrift 的开发复核，不是独立确认。
