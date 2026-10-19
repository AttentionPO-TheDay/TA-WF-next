# v9固定候选的低NAR软证据重排序 v12

用户要求将评价重心转向pAUC，放弃硬min融合，并将v9候选生成与rerank分工。本轮将该建议落实为一次有界缓存实验；新版本受v11结果反馈，Version045/USA均为已观察诊断开发，不是独立确认。只用v9smooth不可变Top1候选与frozen基线，不能新增/替换预测类别。复用direction_iat的3seed×5输入视图，其他模型已在v10/v11形成，没有新训练、新GPU前向或目标reference访问；旧结果封存不改。

关键指标修正：旧纠错vs损伤ROC的负例只有会被v9损伤的候选，FPR不是NAR。新partial ROC用全部frozen正确样本为负例，其中不会改写的负例赋低于所有候选的哨兵分数；正例仅固定v9候选可真正纠正的样本；旧错→其他错不计正负，另完整统计其被接纳数量/每个纠错所需无效改写。故候选阈值的FPR=损伤/全部frozen正确=NAR。TPR分母是固定可纠正候选数，与Recovery分母全部frozen错误不同，二者乘以oracleRecovery对应。仅考核FPR<=.01，报告raw pAUC及raw/.01的归一化面积，不使用McClish把随机映射.5的值冒充平均TPR。也报告0..1% NAR预算下确定性unique-threshold最大Recovery的左阶梯面积/.01，防止插值或拆同分造成虚假低风险收益。sourcevalid可能可纠错正例为0，此时pAUC标为NA而非0。

评分固定8项：v9margin、v11min、等权平均logodds、source可靠性加权logodds、加权均值−.25加权SD、v9源标准化margin+.25源标准化加权均值、同式替换为软方差证据(主候选)、保持v9margin仅打乱辅助软证据对照。logodds=(logit_c-logit_b)/已有温度，无概率截断；15视图相关，因此取加权平均而不累加重复证据，不称校准贝叶斯后验或物理误改概率。sourcevalid固定2160，仅按已有标签算每个seed/view NLL，并以exp(-NLL)归一化为15个固定可靠性权重；用所有sourcevalid行拟合margin/辅助分数均值和populationSD，SD floor .001。不是source-valid独立确认，也没有用query标签优化评分或权重。沿用v10温度，不重新选model/temperature/threshold。方差系数与融合权重均.25，未作超参扫描。

query仅复用缓存logits逐行评分，不能fit任何分布或风险阈值。阈值由query标签枚举仅用于终端pAUC/Rec@NAR1诊断，不能作为部署承诺。源valid所有校准文件先写出并SHA冻结，再统一计算两片终端指标。数据根由configs/datasets.json核对；训练源18553行/102类与原始split不改变。sourcevalid已用于模型选择与温度，不能重划并声称全新评估。

主判据：两片主评分相对v9margin归一化pAUC各增>=.01，Rec@NAR1各增>=.2pp，确定性meanRecovery不下降，接纳其他错误数/真实纠错数不增。主比较只margin_plus_soft；全部次评分报告但不挑最好冒充主。没有全局AUROC成功门槛，不扩大矩阵，不自动部署/整合Teacher。若失败保留v9margin。

预算CPU2线程、10分钟/.5GiB，2partition(valid/query)×2片×3seed×8评分=96状态；0训练/0GPU前向。运行前SHA冻结代码/配置/计划及显式v9/v10/v11缓存，预检三类错误分离/ROC分母/同分保留/手算area边界及与sklearn局部ROC积分独立对照；全部96状态独立复算混淆矩阵、阈值、pAUC，候选集合逐位验证不变、旧baseline前沿精确一致、输入输出hash封存。仅修实现错误保留日志，不按query反馈改参数。

实现修复：首次指标预检因当前NumPy已移除np.trapz失败，尚未拟合校准或计算任何新query指标。改为等价显式梯形求和，所有配置/指标定义/评分/参数保持不变；原freeze与失败日志保留，同配置重新冻结执行。
