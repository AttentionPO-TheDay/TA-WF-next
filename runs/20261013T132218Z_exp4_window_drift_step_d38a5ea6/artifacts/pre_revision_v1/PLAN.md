# 20261013T132218Z_exp4_window_drift_step_d38a5ea6

## 问题
验证图片中exp4_window_drift_step在TemporalDrift上的预注册机制假设。

## 固定协议
3个训练seed（21729、23407、22026）× Day14/Day90/Day270；每单元800次候选更新，目标support仅输入不含标签，query仅在适应结束后评分。完全无伪标签、无GMM、无query梯度；source CE、target entropy/diversity和feature MMD保持一致，学习率固定为0.00025，使用同一已冻结wide+BN checkpoint、support/query split和batch预算。

## 机制
{
  "window_length": 16,
  "slope_span": 4,
  "rule": "scale=clip(1-4*(mean(last4 entropy)-mean(first4 entropy))/16,0.25,1.0)",
  "update": "one scale per target window; first 4 windows use scale=1"
}

## 对照与指标
每个单元以同seed的frozen模型为配对对照；报告accuracy、Macro-F1、负适应率、纠错率、改变率、实际更新数、反向耗时、query推理时延和峰值显存。主筛查线为Macro-F1超过73.504%固定低学习率参考且负适应率低于2%；不因未达线而删除结果。

## 边界
仅使用已观察TemporalDrift开发数据；WTT-Time、AWF及Version/Network/Behavior不在本批次。
