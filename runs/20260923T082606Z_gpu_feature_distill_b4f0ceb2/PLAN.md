# 20260923T082606Z_gpu_feature_distill_b4f0ceb2

问题：同一融合教师的window中间表示能否比教师packet表示和普通CE更有效地训练packet-only学生，并改善valid及时间日期泛化？

状态：frozen，训练前登记。

问题：单纯输出 logit 蒸馏阴性之后，融合教师的中间表示是否能帮助只接收 packet 的学生？本轮为新的预定机制，不改动上轮结论，也不加载上轮 checkpoint。教师与学生基础骨干显式改写自本项目 `20260923T080156Z_gpu_train_only_token_distill_548c9e47`，行为通过前向/梯度和重载核验。

数据：复用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 固定 source/valid/Day14/30/90/150/270 行号；路径来自 `configs/datasets.json`。有符号时间戳只取前 5000 个方向，50/250 window、partial 中和 (0.5,0)，窗口统计仅由 full source 计算。source 标签用于训练教师和学生；valid 标签分别用于教师、学生按 macro-F1 选择最早最佳 epoch；未来标签仅用于学生模型选定后的 TemporalDrift 开发评分。

教师：同前轮 219494 参数 packet+window 融合网络，每 seed 从头初始化，AdamW lr0.001/weight decay0.0001、batch128、45 epochs。选出的教师只在 source 上输出其 packet 分支 128 维与 window 分支 128 维激活；两种 target 分别按 source 每维均值与标准差归一化，标准差下限0.001。教师不对未来样本生成 target。

学生：同一 packet-only 159078 参数 CNN 与分类头；训练期另设 128→128 线性投影，推理时不调用。三条件 `ce`、`align_packet`、`align_window` 共享逐 seed 模型与投影初始化、batch 顺序、45 epochs 和 valid 选模。CE 条件仅交叉熵；两个对照都使用 `CE + 0.5*MSE(学生投影特征, 教师标准化特征)`，分别以同一个融合教师的 packet 或 window 分支为目标。CE 条件的投影层被实例化但没有梯度，训练可用参数较少；两特征条件完全匹配。学生推理只接收 packet、不生成 window、不调用投影层或教师。

判定：`align_window` 相对 CE 及 `align_packet` 均需 valid F1 均值正、至少2/3 seed正，并在五未来日期平均差正且至少4/5日期均值正，才称训练期生成 window token 的候选收益。训练 accuracy、MSE 和绝对日期分数一并报告；未来日期不选模。若只改善拟合或仅优于 CE，不能归因于 window 特定监督。

预算/停止：GPU 0 单卡，三 seed×一个融合教师×45 epochs，加三 seed×三个学生×45 epochs；1200 秒上限。配置非 frozen、CUDA 不可用、已有 checkpoint/最终预测、固定清单异常、source/valid 方向交叉或非有限输入时停。原始数据只读；外部 WTT-Time/AWF 不访问，所有日期结果只作开发证据。
