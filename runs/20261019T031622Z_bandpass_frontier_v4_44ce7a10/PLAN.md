# 安全—纠错前沿 v4

用户2026-10-19授权启动上一轮提出的90任务实验。数据、实现、指标和预算随config.json冻结后执行。

问题：在相同NAR预算下，带通/软权重/几何可靠性是否提高纠错？先分析旧候选上限、排序曲线、随机接受线，再运行三种路由。四组A无路由（复用v3 soft_neighbor_lora）、B硬带通、C软带通、D软带通×几何。固定LoRA r4、源邻域软目标、KL/原型/关系/源identity损失、400步×64、source32、Adam0.00025，最终步评分。B/C共享无标签reference H20/H90；平滑温度max(.1带宽,.02)。D追加冻结PMR及绝对Mahalanobis可靠性。全权重detach，目标损失按B归一化，关系项按weighted class centers计算后乘mean(w)，不改变source项。A必须通过同批无权重损失、全query冻结前向与旧文件hash一致性核验。

数据与权限：既有10诊断切片，Version045–048、Network DE/JP/SG/UK/USA、Behavior；保持1024无标签reference与query的原隔离。Hostname/URL只用于原清单分组。source18553、valid2160；valid已参与底座选模，不能称新确认。无query梯度、统计拟合、方法选择或实际阈值校准。query标签只进入终端评分/事后oracle和曲线分析。本批为已观察诊断开发，非未见盲测；WTT/AWF不新增评分。

输出判据：所有候选采用同一冻结几何+同类归一化logit裕度增益评分。几何由冻结特征计算，避免适应后自证。每个适应候选在source valid上以预定grid分别校准1%/2%NAR运行点（>=10变化且正净纠错，最大恢复；不可行全回退）。另以query标签仅事后绘制完整分数阈值曲线，ties不可分；与同法接受的reference-TENT、direct rerank、无路由LoRA/Residual比较。事后按目标NAR挑点仅是诊断可达前沿，不宣称可部署安全。

主判据：D相对A，在主score诊断NAR<=1%时Recovery提高>=0.5pp，三seed均正，至少两个因素正且余下>=-0.5pp。与所有非零对照的最强前沿比较单列，不能只选frozen作结论。B/C、2%NAR、相同Recovery下NAR、source-calibrated运行点均次要。各因素内部切片等权，三因素等权，seed42/100/2026报告均值与SD和逐因素；不把row×seed作为独立样本，不以3seed宣称显著性。完整失败/全部回退保留。

若C或D在上述1%前沿相对A达到>=.5pp且三seed正，触发预定batch内权重打乱阴性对照，最多30任务；选满足条件且增益更大的软arm，同增益优先D。该选择明确使用诊断反馈且只作归因，非独立确认。否则不加跑。打乱保留每batch权重分布和原样本目标，只破坏对应关系。

预算：GPU0/1/2分别seed42/100/2026，90新适应，400步；最多30条件性打乱、总6GPUh/2h wall/4GiB；单任务600秒，预算到限停止保留。不得干预无关GPU任务。源/架构preflight和预定全query checkpoint重载审计不做方法选择。产物：原始/运行点主表、风险—纠错曲线与随机/oracle参照、因素表、路由/有效样本/梯度日志、完整checkpoints与SHA审计。代码冻结后不得因新结果调参。
