# 候选纠错与验收 v3 结果

630/630评分状态完成（270原始状态、360派生验收状态）；210次适应训练，TENT两种推理共享一次优化。全部240个适应checkpoint全query重载与630指标复算通过，30个frozen重算；拒绝行逐位回退核验通过。

描述性最高的严格reference方法：tent_bn_reference/raw，F1 43.864±0.290%，ΔF1 +1.983pp，NAR 8.513%，Recovery 8.546%。不是根据该值追加训练或选取checkpoint。

同预算reference-only TENT：F1 43.864%，ΔF1 +1.983pp，NAR 8.513%。标准TENT使用query批统计，单列，不混称严格reference对照。

预注册开发判据通过 0/16 项；完整裁决见artifacts/registered_decisions.json，不能以某个有利切片取代整体判据。

完整表见 [主表](table_main.md)。逐seed/slice、逐因素、配对目标差异、验收计数、源校准可行性均在artifacts中；图为固定样本和共享坐标实测表征，不保证语义恢复。

本轮hard也含KL项，因此与v2原型hard不是单一soft开关对比；本轮hard/soft_teacher/soft_neighbor之间才是匹配损失的监督信号对比。v2历史控制不算本轮新重复。

限制：source valid曾用于底座选模，校准NAR约束不等于目标保证；源库邻居不能保证跨域正确，Top5候选会漏掉部分真类；目标标签仅评分，邻居证据由固定源库计算，无query批次拟合。方法受到既往Temporal、多因素及外部v4反馈影响，本轮仅已观察诊断开发，不是独立确认。

补充 [事后直接重排对照](table_posthoc_direct.md)：使用同一固定源邻域证据、无需任何适应，检查复杂对齐是否优于直接推理。这是配方冻结后补充的探索性诊断，不参与本轮训练或阈值选择。

验收可行性：三个seed在source valid均无满足预定风险门槛的邻域验收阈值，所以evidence_accept在目标上采用预注册的全部回退；其零NAR与零Recovery是分支不可用的结果，不能算成功治理。valid_entropy是另一个可用分支，阈值按源验证分位数冻结。

计数说明：cells中的tent_bn_reference的steps=0与parameters=0表示该保存状态未另做优化且推理时参数已冻结；其训练明确复用对应tent_bn的400步及同一BN affine参数，不能解读为无训练获得效果。630状态中180个evidence_accept是上述不可用分支的冻结回退，不是180次新适应。
