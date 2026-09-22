# Execution record

日期：2026-09-16（Asia/Shanghai）。工作目录始终为 `/home/rbf/TA-WF-next`。

- 先完整读取项目协议、donor 与表示诊断正式 artifact、checkpoint metadata/history、split/hash 和模型定义；随后在任何 target-date 拟合或评分前冻结 `RECOVERABILITY_PLAN.md`。
- 先运行内容级隔离审计并生成固定 manifest。恢复中断后按已记录 SHA-256 复核并复用这些 artifact，没有重跑审计、重抽 support 或改写计划。
- DF G-source 在中断前已完整生成，恢复时只校验复用；仅重新执行未落盘的 VarCNNDirection G-source。两者均只做冻结前向和 CPU logistic regression，不存在 backbone optimizer/backward。
- 依次执行两个 backbone 的 Day14/90/270，每个日期一次冻结特征抽取，使用 manifest 内 3/10-shot×3 seeds；每日期全部预测固定后才评分。
- 执行 `summarize` 生成 144 行 `metrics_long.csv` 与 `summary.json`；独立 verifier 重算所有 accuracy/macro-F1 并检查 query、hash、权限、训练计数和临时文件。
- 验证命令：`py_compile` 通过；实验专用单元测试 3 passed；`verify_recoverability.py` passed（36 evaluation artifacts、3,319,104 method prediction rows、0 errors、0 convergence warnings）。
- 新 backbone 训练/微调：0；checkpoint 写入：0；外部数据/TTA/表示更新/DNNF/TFAN：0。
