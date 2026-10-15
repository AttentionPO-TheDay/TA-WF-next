# 20261014T003143Z_tta_subspace_restriction_f86ece7f

方案B：将无标签适应限制在两个参数子空间，判断负适应主要来自Wide生成器支路还是Transformer注意力投影。

条件一 `wide_bn_only`：仅generator.packet_conv、generator.stages、generator.to_features及其BN可训练；其余Transformer主干、投影、位置、FFN、final norm、classifier冻结。

条件二 `attention_bn_only`：仅head self-attention投影和生成器BN可训练；生成器卷积/输出、投影、位置、FFN、final norm、classifier冻结。

TemporalDrift已观察开发集，3 seed×Day14/90/270，每单元800次更新，Adam lr=0.00025，source CE + target entropy/diversity + feature MMD；support无标签，query仅最终评分。复用既有精确split，不重划分；无伪标签、GMM和query梯度。

报告Macro-F1、accuracy、负适应率、纠错率、可训练参数量、更新时间和显存，所有18个单元保留。WTT/AWF关闭。
