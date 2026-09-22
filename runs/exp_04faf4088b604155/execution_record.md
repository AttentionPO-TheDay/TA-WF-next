# Execution record

日期：2026-09-17（Asia/Shanghai）。工作目录始终为 `/home/rbf/TA-WF-next`。

- Provider recovery 开始时先读取 `STATUS.md`、`PROTOCOL.md`、`HANDOFF.md`、本实验冻结计划及 donor artifact，并检查工作树与现有文件。
- 未重跑已通过的 DF embedding cache。对现有 VarCNNDirection source/current cache 做 SHA-256、shape、dtype、finite、逐行范数、row uniqueness 和 source geometry 检查，全部通过。
- 仅执行缺失阶段：6 个 backbone×date freeze 调用，共生成 18 个 seed unit；随后结构检查通过并生成 firewall seal。
- seal 后执行一次 score，生成重采样、site、support-only、错误归因、probe 和汇总 artifact。评分入口先复核全部 seal 哈希，并核对 donor 正式预测。
- 新增只读 verifier `code/verify_diagnostic.py`；`py_compile` 与 verifier 均通过，`integrity_check.json` 记录 34 个 sealed 输入、18 个 frozen unit 及全部输出行数/哈希。
- 新 backbone 训练/微调：0；checkpoint 写入：0；外部数据、TTA、表示更新、聚类、阈值修改、query-driven 选择和新方法设计：0。
