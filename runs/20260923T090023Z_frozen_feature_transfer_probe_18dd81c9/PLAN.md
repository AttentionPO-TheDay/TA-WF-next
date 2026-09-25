# 20260923T090023Z_frozen_feature_transfer_probe_18dd81c9

问题：冻结已选教师和packet学生后，window特征的valid类别判别力与从学生packet表示的可预测性是否同时成立？

状态：frozen，评分前登记。

问题：前两轮训练期蒸馏未稳定受益，是因为融合教师 window 分支本身缺乏类别判别力，还是 packet-only 学生的现有表示难以恢复它？本实验只作诊断，不尝试新蒸馏权重或方法选择。

模型来源：显式复用本项目 `20260923T082606Z_gpu_feature_distill_b4f0ceb2` 中已由 valid 选定、完整性核验通过的三 seed 融合教师及普通 CE packet 学生 checkpoint；不加载旧项目模型。记录每个 checkpoint SHA-256。既有教师/学生已在同一 valid 上选模，本轮结果仍是开发证据，不能视作独立验证。

数据：复用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 的 source 2040/valid 510 固定行号，通过 `configs/datasets.json` 定位只读原始数据。仅前 5000 个方向；50/250 window、partial 中和 (0.5,0)，full source 统计标准化。仅 source 标签用于线性类别探针拟合，valid 标签只用于评分；没有未来日期访问或训练。

固定分析：三 seed 各提取教师 packet/window 两个 128 维特征及 CE 学生 128 维 packet 特征。对 source 特征做 source-only 每维标准化（std<0.001 时置 1）。用同一 alpha=10 的带截距多输出 ridge，在 source 拟合 102 类 one-hot：教师 packet、教师 window、教师二者拼接及学生特征四种输入；valid 报 accuracy/macro-F1。另以学生 source 特征拟合教师 packet/window 特征，在 valid 报相对 source 均值预测的方差加权 R²和 MSE。把重建的教师特征送入先前对应的类别探针，比较真实和重建特征的 valid F1。类别探针和重建器都仅用 source 拟合，不用 valid 选参数。

解释边界：真实 window 探针若强而重建 R²/F1 低，提示当前学生表示/线性读出难以转移；真实 window 探针若弱，提示这一逐维目标本身可能不合适。线性重建失败不证明非线性学生无法学；高 R²也不保证任务分数改善。教师已通过 valid 选模且数据已多次用于开发，不能作新确认或显著性推断。

预算：GPU 0 用于冻结特征提取，CPU 最多 4 线程做闭式 ridge；三 seed、source/valid 两角色、180 秒上限。配置非 frozen、checkpoint/清单散列不符、源与 valid 方向重叠、类别或形状异常、非有限输入时停止。WTT-Time/AWF 保持关闭；原始数据只读。
