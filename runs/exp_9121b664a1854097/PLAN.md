# exp_9121b664a1854097

独立执行 `exp_6238dacf9aa142cc` 已冻结的 TemporalDrift screening 协议。旧 experiment 只提供明确指定的 PLAN、RUN_COMMANDS 和其引用的冻结 artifact；不恢复其 Job、thread、planner、checkpoint 或运行状态。

协议内容逐项见 `execution_requirements.md`。研究问题、数据与标签权限、模型、训练预算、A/B/C/D/E 定义、评价日期、指标和停止条件与旧实验当前 PLAN 的 v4 冻结要求一致。所有新 checkpoint、日志、预测、指标和报告只写入本目录。

状态：frozen；获用户明确授权执行最多两次 GPU 训练与冻结评价。
