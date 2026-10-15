# 执行修订记录

首次执行在第一日期的随机选择条件进入模型前发现随机门控张量为 float64，导致与 float32 线性层 dtype 不一致；该执行没有完成任何结果汇总，也没有写入有效预测。将随机门控显式转换为 float32 后，使用相同配置、数据划分、checkpoint、种子、适应池、query 池和更新预算重新执行。最终 36 个预测文件和 27 个适配器 checkpoint 通过 postrun_audit；首次失败 traceback 保留在 `artifacts/failure.json`。
