# 原型细化 v6：两切片有界开发

问题：v5 momentum_proto较batch momentum有效，而joint无增益；六个已观察单元原型类间距离收缩约9–10%。本轮检验移动loss锚点与几何软目标的独立作用，并检验锐化几何或保持源类别半径是否改善纠错—风险前沿；这些只是机制假设。

本轮已受v5目标反馈影响，属于diagnostic development，不是盲测或确认。数据角色承接MULTI_FACTOR_DIAGNOSTIC_PROTOCOL.md。Version045为可归因公开诊断子集；network_USA仅称跨地域节点，不能归因具体丢包率/时延。完全沿用冻结1024无标签reference/query分割、精确源隔离，不重抽分区。

权限：源模型只用18,553纯净Temporal train；三个源训练seed42/100/2026。源标签用于既有源原型/邻居，源valid标签仅终端评分。reference标签不向优化API提供。query输入只用于独立逐样本终端推理，标签仅指标与注册事后曲线，绝不更新EMA/原型/记忆库。WTT/AWF不访问、不评分。新版本方法比较允许使用本轮开发评分，但不允许边跑边追加/改超参数。

六组各2切片×3源训练seed，共36次400步新适应：anchor_only（仅移动loss锚点）、target_only（动态几何目标＋固定源loss锚点）、fixed_geometry（固定源几何目标＋固定loss锚点）、sharp_proto（v5 proto几何温度.1→.05）、radius_proto（v5 proto以原始锚点均值作参照中心，恢复每类源中心半径；变换后整体均值不保证不变）、sharp_target_only（target_only＋温度.05，预定主候选）。每组的精确开关在config.refinement_v6.arms，所有其余配置与v5相同。

共同：Q/V LoRA rank4，4096参数；骨干/BN/分类头冻结；Adam lr.00025，400×64，源identity32行/步，完整102类软目标，目标混合.25，源锚定.5，正值动量权重最低.05，batch mean损失，无输出旁路/重排。原型/权重/几何目标detach，优化后按前向观察更新reference状态。源模型seed不同，适应初始化与批流固定20261019；不是三个适应随机流。

控制：复用v5 full_static、batch_momentum、momentum_proto和v3 strict reference-TENT的已审计预测/信号；不重复训练已完成控制。源域预检要求legacy_control与v5实现loss、梯度、目标及一步reference状态相等，零初始化相同批形/同no_grad模式预测完全一致，并验证锐化熵、radius约束、独立开关与骨干冻结。仅通过预检后启动目标适应。

指标：逐seed/slice Accuracy、Macro-F1、相对frozen ΔF1、NAR=原先正确变错/原先正确、Recovery=原先错误变对/原先错误；均值及样本SD。沿用同一个frozen common acceptance score的事后Recovery@NAR1/2和NAR@Recovery2/4诊断。事后阈值用query标签定位，不构成本轮部署规则，不允许反向写入预测。

主候选sharp_target_only预定通过标准：两片均比v5 proto平均F1增加>=.20pp，raw NAR增加<=.20pp，同score Rec@NAR1不下降，且逐源seed相对frozen增益为正。secondary sharp_proto/radius_proto分别按同标准报告，不替换主候选。其余三组用于机制归因，无主候选资格。扩展资格另需主候选两片ΔF1均>=3pp；本run不启动扩展。结果全部保留，不以最佳seed/arm遮蔽阴性。

预算：GPU0/1/2分别源seed42/100/2026，并行36任务；最多1h管线/3 GPUh、2GiB新产物、每cell600s；不停止其他实验。无新源模型训练。若预检失败保留证据并仅修实现错误；目标失败不追加搜索，预算内中断恢复只续未完成任务。所有36 checkpoint全query及source-valid重载，独立混淆矩阵复算指标、纯净底座/状态尺寸/输入输出SHA核验后交付。
