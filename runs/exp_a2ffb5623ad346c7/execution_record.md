# Execution record

日期：2026-09-16（Asia/Shanghai）。工作目录始终为 `/home/rbf/TA-WF-next`。

- 在任何新 current-query 预测/评分前完整读取项目协议和指定 donor 证据，核对 checkpoint、split、manifest、隔离审计、G-source 与 donor evaluation 哈希；随后冻结 `HISTORY_RETENTION_PLAN.md`（SHA-256 `909850d12eed764d0f881bd7d61860d2f06d189ebdcaf200f9a69ba4d4959561`）。
- `preflight` 完整解析所有指定大型 JSON/CSV/joblib 与 checkpoint/history；18 个 manifest 配置、36 个 donor artifact、144 行指标均通过，0 errors。
- `select-source` 只用 source-holdout 伪 support 与 official source validation，在两个 backbone×2 shots×3 seeds 共享选择 `alpha=0.25`、`lambda=0.1`；记录 `query_dates_accessed=[]`。
- 六次 `evaluate` 均在 CPU 做冻结前向；每个 backbone×date 先固定全部 6 个配置的新预测，再访问 donor query truth 评分。没有 GPU、optimizer/backward、backbone 更新或 checkpoint 写入。
- 验证：`py_compile` 通过；实验专用单元测试 3 passed；独立 verifier passed（36 evaluations、4,978,656 prediction rows、0 errors、0 nonconverged shrink fits）。
- 关键 SHA-256：`source_selection.json` `70c42b74a360493432eccdcfc83128b7d97374f036cf404c12c125cce80f357b`；`summary.json` `c6867b8b14c8590a96a014703de1525173167c227c3ee0a5214db3c05a01f468`；`integrity_check.json` `acc699e61fdfa6d157bffc20a0154ccc17f884c923975346a49d47a3d79594c5`；`posthoc_repeatability.json` `e1137163122ccdb6c37346c0426a6a547678aebbef754669e2428d4b95d108a8`。
- 新 backbone 训练/微调：0；checkpoint 输出：0；外部数据/TTA/adapter/attention/memory/DNNF/TFAN/多日期连续适应：0。

执行命令类别：

```text
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python -m py_compile ...
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python -m unittest discover -s runs/exp_a2ffb5623ad346c7/code/tests -v
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python runs/exp_a2ffb5623ad346c7/code/run_history_retention.py preflight
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python runs/exp_a2ffb5623ad346c7/code/run_history_retention.py select-source
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python runs/exp_a2ffb5623ad346c7/code/run_history_retention.py evaluate --model {df,varcnn_direction} --date {day14,day90,day270}
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python runs/exp_a2ffb5623ad346c7/code/run_history_retention.py summarize
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python runs/exp_a2ffb5623ad346c7/code/verify_history_retention.py
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python runs/exp_a2ffb5623ad346c7/code/analyze_repeatability.py
```
