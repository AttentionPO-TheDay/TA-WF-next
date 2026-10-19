# 组合实验结果

240/240 新单元完成；全部checkpoint全query重载、基座锁定、指标复算、拒绝行逐位旁路审计通过。

完整结果见 [组合总表](table_combinations.md)、[因子效应](table_factorial_effects.md)、[历史参考](table_historical_controls.md)；逐切片、逐因素与240单元CSV见artifacts。

本批描述性最高F1：single_ungated_lora，F1 42.592±0.138%，ΔF1 +0.712pp，NAR 2.285%。这是已观察诊断结果中的比较，未据此返调或新增外部评分。

主效应：
- GMM4-Single: F1 -0.007pp，NAR -0.013pp。
- Routed-Ungated: F1 -0.663pp，NAR -2.128pp。
- LoRA-Residual: F1 +0.040pp，NAR +0.140pp。

图为实际提取的Transformer后128维特征；LoRA前Token不变。t-SNE是描述性图，必须结合visual_diagnostics.json的高维类心距离、邻近准确率和方差判断，不能据低维聚集宣称语义恢复。

样本高熵或距原型远并不证明物理上不可恢复。门控只保证被拒绝行原预测不变，接受行仍可能误杀；GMM源域分量不保证对应真实页面模式。W2采用点高斯到分量高斯，协方差影响软分配，不能宣称估计了测试样本协方差。

输入、代码、完整配置、源拟合产物的SHA256见artifacts/freeze.json；既往外部v4与诊断v1均已影响本轮方案，只作开发证据。

交付补充核验：本轮30组frozen与旧v1共49条预测差异，逐组logit偏差见 `artifacts/frozen_reuse_comparison.csv`。本轮为使LoRA生效统一禁用MHA fastpath并从加载时冻结底座参数；不同attention数值内核导致少量边界预测变化。所有本轮ΔF1/NAR/Recovery均相对本轮同后端重新计算的frozen，不用旧预测替代，旧表仅历史参考。门控接受/拒绝的正确率、熵、距离与计数见 `artifacts/routing_diagnostics.csv`。`gateonly_counterfactual.csv` 仅事后计算推理旁路会同时压掉多少误杀和纠错，不是返调或额外训练的候选。5组图各PNG/PDF/SVG已检查。

本轮判读：四个routed组合在全部120单元的分类预测均与本轮frozen一致。 源训练样本拟合得到的H95约为0.060–0.097 nat，明显偏向极自信样本；加上推理硬旁路，NAR=0同时Recovery=0，不能作为成功治理证据。用已拟合训练行校准熵阈值可能过于乐观；若后续改用源域保留验证样本校准，必须另立版本、记录本轮目标反馈并预先冻结，不能覆盖本轮结果。

GMM4对照single的收益需按完整因子效应判读；本轮没有支持多原型更优。单原型LoRA相对残差的F1增量较小且负适应更高，不足以声称LoRA支配残差方案。无门控方案虽有F1净增益，但NAR>0；本轮未建立绝不增加误杀的机制。所有组合共用源锚恒等约束，因此不能将与旧v1的差别单独归因于GMM或LoRA。
