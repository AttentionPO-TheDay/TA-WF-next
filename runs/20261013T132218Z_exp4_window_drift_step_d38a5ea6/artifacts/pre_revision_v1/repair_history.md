# MVI 审计恢复记录

本轮 45 个新适应任务和 27 条历史参考均已完成。首次 postrun 审计失败两次：

1. 历史参考的负适应、纠错和 changed_rate 使用来源 run 的 frozen 预测，而本轮统一审计使用 MVI run 的 frozen 预测，配对基准不一致。原始文件保留在 `artifacts/scores_pre_repair.json`；历史参考指标已按本轮 frozen 预测重算。
2. 新任务保存 logits 与 checkpoint 重载 logits 存在 CUDA 浮点差异，代表 batch 的 argmax 保持一致。原始新结果保留在 `artifacts/scores_pre_checkpoint_repair.json`；最终评分预测采用已保存 checkpoint 重载后的结果，指标随之重算。恢复审计要求 logits atol=1e-2 且代表 batch argmax agreement≥99%。

修复只涉及配对记录、checkpoint 复核和指标审计；未重新训练、未改变五个新条件、输入划分、标签权限或目标阈值。`artifacts/failure.json` 保留首次失败证据；`artifacts/audit.json` 是恢复审计结果。
