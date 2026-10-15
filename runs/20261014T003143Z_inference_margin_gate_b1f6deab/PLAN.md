# 20261014T003143Z_inference_margin_gate_b1f6deab

方案A：推断期安全门控能否在不重新训练、不反向传播的情况下，利用Frozen源模型置信度保护高置信样本，降低Exp-1适应的负适应，同时保留纠错收益？

复用已完成TemporalDrift的3 seed×Day14/90/270 query预测：Frozen源模型与no_pseudo_fixed适应模型的logits和完全相同query rows。query标签只用于开发评分、阈值曲线和预注册选择，不进入门控规则。

Margin门控：margin_src > tau时取Frozen预测，否则取适应预测。熵门控：H_norm < epsilon时取Frozen预测，否则取适应预测。扫描固定margin 100点[0,0.99]与entropy 101点[0,1]。选择均值负适应率<2%约束下最大均值Macro-F1，平手时负适应率更低、再平手阈值更小；若无可行阈值，报告最小负适应率候选。

仅TemporalDrift已观察开发数据；WTT/AWF及其他因素关闭。
