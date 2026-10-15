# 结果记录修复

2026-10-13：54 个低学习率损失消融训练任务均完成。原始执行在 `postrun_audit.py` 首个断言处失败，原因是 runner 从比较 run 复制了该 run 的全部 90 条结果（其中包含上一轮 54 条消融结果），而本实验审计和 PLAN 预定的历史参考应为 36 条：`frozen`、`global_entropy_adapter`、`stepwise_rollback`、`proteus_port` 各 9 条。原始 144 条文件保存在 `artifacts/scores_pre_repair_144_rows.json`，`artifacts/failure.json` 保留。

恢复操作只从原始结果文件筛选上述 36 条历史参考，并保留本轮 54 条 `origin=new` 结果，生成 `artifacts/scores.json` 共 90 条；未重新训练、未修改冻结代码、输入、预测、checkpoint 或标签权限。随后使用原始 `postrun_audit.py` 重新核验，并重新生成汇总。该修复属于记录/审计错误恢复，不是方法或超参数调整。

审计恢复结果：`artifacts/audit.json` 为 passed=true；随后使用保存的预测重新生成 `RESULTS.md` 和 `contrasts.json`。
