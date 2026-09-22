# 可继续执行的命令

v3 split 保持不变；当前评价使用 v4 浅层逐样本 mask 和 E 同层全局对照，旧深层 D 已由训练前 source 审计修订。`prepare` 不应重跑（入口会拒绝覆盖）。`audit-length` 和 CPU `smoke` 的 v4 证据已保存，也不应覆盖重跑。GPU 空闲且用户继续训练时，在 `/home/rbf/TA-WF-next` 使用同一显式 PyTorch环境依次执行：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py train --model df 2>&1 | tee runs/exp_6238dacf9aa142cc/logs/df_train.log
CUDA_VISIBLE_DEVICES=<free_gpu> /home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py evaluate --model df 2>&1 | tee runs/exp_6238dacf9aa142cc/logs/df_evaluate.log
CUDA_VISIBLE_DEVICES=<free_gpu> /home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py train --model varcnn_direction 2>&1 | tee runs/exp_6238dacf9aa142cc/logs/varcnn_direction_train.log
CUDA_VISIBLE_DEVICES=<free_gpu> /home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py evaluate --model varcnn_direction 2>&1 | tee runs/exp_6238dacf9aa142cc/logs/varcnn_direction_evaluate.log
```

运行前再次执行只读 `nvidia-smi`。不得改变 split、seed、epoch、reference 或 D 规则；若实现错误需修订，保留原 artifact 并登记版本和原因。
