# 小范围动量原型与批内软对比 v5

用户2026-10-19授权：取消静态硬旁路，尝试批次动量权重与动态原型、批内对比联合，先Version045与Network USA；以两切片ΔF1>=+3pp为全量扩展门槛。目标不是结果保证，不能跑到达标才停止。

本轮30新适应：5arm×2slice×3seed42/100/2026。full_static为完整102类软目标的匹配静态对照；batch_momentum加入持续正权重；momentum_proto加入EMA原型；momentum_contrast加入批内软对比；momentum_joint联合。所有arm均从纯净源18,553行模型重新加载并只更新rank4 Q/V LoRA，400步×64，源anchor32，Adam .00025，clip1，最终步固定评分。源BN/分类头/底座完全冻结。

统一去掉旧Top5目标截断，源邻域soft targets保留102类，新增full_static与batch_momentum分离类别支持变化/路由作用。基础KL(T2)、原型pull .1、关系拓扑1、源identity1均保留。原型arm以.25混入源锚定动态原型分布；动态原型等于.5源中心+.5目标EMA中心，软类质量调节.95动量，防止低支持类别瞬时漂移。对比项.1、temperature .1，positive targets为detached q_i·q_j，排除对角；不会把同一条样本自我匹配当作成功聚类。

批次动量权重由无标签reference概率EMA的熵与batch EMA一、二阶矩给出，权重在[.05,1]，不硬筛除任何样本，不对输出做硬回退。feature/probability memory及动态原型只使用reference，在每arm/seed/slice重置；prototype/q使用上一状态，当前batch优化后再更新memory，防止同一步自证。source prototype anchoring是训练正则，不是推理旁路。状态/漂移/正样本质量/熵/梯度必须记录。

数据和权限：沿用诊断v1两切片原1024reference/query，reference优化接口无标签；源labels只用于已有源bank/anchor；query输入不进入优化、EMA、prototype或统计拟合，标签仅终端评分与预定事后曲线。source valid仅终端源性能，不能选step。历史v3/v4对照仅复用并核验SHA，不算新重复。数据已观察，全部是诊断开发，不是新确认；不新增WTT/AWF。

输出：每slice/arm三seed均值SD的MacroF1/Accuracy/ΔF1/NAR/Recovery、源valid保留、参考无标签损失和原型漂移、共同预定score的事后NAR—Recovery前沿（score复用v4冻结几何/同类归一化logit裕度增益，1%/2%预算）。该曲线用query标签事后定位，只作诊断，不作为部署安全承诺；主raw输出始终不用任何输出筛选。

扩展判据：预定joint在两slice的平均ΔF1均>=3pp，6个seed×slice均正，且两slice的诊断Recovery@NAR<=1%不低于full_static。报告vs匹配对照与同预算ref-TENT，不把仅超过frozen视为机制优越。其他arm仅消融，不事后改主arm。未达标停止，不在query上调超参数重跑。达到门槛也只登记扩展资格，下轮按明确预算冻结全量实验；不自动扩成240任务。

资源上限：GPU0/1/2分seed，不影响现有无关进程；30任务，400步，wall1h，3GPUh，2GiB，单任务600秒。源/梯度preflight在真实评分前完成；全query checkpoint重载和独立指标/前沿审计后交付。完整代码/源bank/纯净权重/数据cache/config/plan冻结SHA。
