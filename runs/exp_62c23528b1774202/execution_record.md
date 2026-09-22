# Execution record

日期：2026-09-17（Asia/Shanghai）；工作目录始终为 `/home/rbf/TA-WF-next`。

- 恢复任务后先执行 `git status --short`、`git diff --stat` 并核对实际文件；目标 run 当时不存在，未假定前一 provider 留有可续写产物。
- 完整读取 `AGENTS.md`、`STATUS.md`、`PROTOCOL.md`、`HANDOFF.md`，以及 `exp_b471517a3e6f41e7`、`exp_04faf4088b604155` 和 `exp_376fca9354214097` 的冻结计划、正式结果、机器 artifact、integrity 与执行记录。`input_manifest_v1.json` 保存全部实际依赖路径与 SHA-256。
- 在任何新 B/C/D/消融预测和评分前冻结 `METHOD_PLAN_v1.md`（SHA-256 `7575364104ec920fce5d0cf0760198642f5c4eb67de63d257576bba08107b527`）。运行 `py_compile` 与实验单元测试，2 tests passed；随后执行 `preflight`。
- 只读复用 `exp_04faf4088b604155` 的 embedding/cache 与 source geometry，在 CPU 上生成两个共享历史包及 18 个未评分 unit。seal 前核对 common query/support、形状及 B 与既有 multiprototype probe 的逐预测一致性。
- 生成 firewall seal（SHA-256 `84e682ed20428556030c04cf45a210755bef1d6a134103cd602f8ad7c8982e7b`）后才执行 `score`，读取既有 A evaluation 中的共同 query truth 和正式 A prediction。
- 独立 verifier 重算 90 行指标和 2,042,310 条 prediction，检查 25 个 sealed 文件、正式前三条 support、B prior probe、共享库与预算；`integrity_check.json` SHA-256 `bebbce87eb0aa61b24724abcdb2ecab30d70886e4b63f9e13066807e2111bd90`，`passed=true`、0 errors。
- 关键执行命令：`/home/rbf/TA-WF/.venv/bin/python -m py_compile ...`；`-m unittest discover -s runs/exp_62c23528b1774202/code/tests -v`；`run_region_relevance.py preflight|freeze|seal|score`；`verify_region_relevance.py`；`git diff --check`。
- 新 backbone 训练/微调、checkpoint 输出、GPU 工作、表示更新、TTA、外部数据、10-shot 算法输入与后续 Job：均为 0。
