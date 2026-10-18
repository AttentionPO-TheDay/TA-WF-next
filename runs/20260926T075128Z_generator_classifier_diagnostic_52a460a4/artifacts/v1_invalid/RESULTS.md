# 实验结果

v1因关闭requires_grad改变单条source预测的CPU舍入结果而作废，已归档；v2保留原参数标志，在inference_mode禁用梯度，统一重跑所有探针。方法、lambda和裁决规则未变。详见PLAN的实现修正记录。
