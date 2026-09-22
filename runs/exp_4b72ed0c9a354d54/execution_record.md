# Execution record

日期：2026-09-16（Asia/Shanghai）。工作目录始终为 `/home/rbf/TA-WF-next`。

## 顺序与命令

1. 完整读取 `AGENTS.md`、`STATUS.md`、`PROTOCOL.md`、`HANDOFF.md`、donor `PLAN.md`/`RESULTS.md`/`execution_requirements.md`/`config.json`、两份 history、v3 split、v4 layer/content audit 和模型定义。
2. `sha256sum` 核对两 checkpoint、split、v4 layer spec；CPU hook smoke 核对 checkpoint metadata/history best 与层 shape。第一次临时 hook 回调错误地返回 shape 并替换输出，立即失败且未写 artifact；改为只记录/返回 None 后通过。没有修改 checkpoint。
3. `/home/rbf/TA-WF/.venv/bin/python scripts/run_representation_diagnostic.py audit`：只读 source train/valid，写 source coverage audit；未读 future。
4. `py_compile`、`PYTHONPATH=src ... -m unittest discover -s tests -v`、diagnostic `smoke`、`scripts/experiment.py check`、`git diff --check`。
5. 分别运行 `... run_representation_diagnostic.py source --model df` 与 `--model varcnn_direction`。这是冻结 GPU 前向与 CPU logistic regression；不存在 backbone optimizer/backward。
6. `... summarize_representation_diagnostic.py source` 写完整 source 矩阵、C 选择记录和 `SOURCE_RESULTS.md`。新浅层 global/unordered 预测与 donor E/D 精确一致后，运行 `... run_representation_diagnostic.py freeze`，冻结 source artifact 和 linear model hashes。
7. 将 config 状态切为 `future_frozen` 后，分别运行 `... run_representation_diagnostic.py future --model df` 与 `--model varcnn_direction`；每个命令固定评测 Day14/30/90/150/270。之后只汇总，不改配置。

执行完成后，两个实验专用入口和专用测试从共享 `scripts/`/`tests/` 归档到本 run 的 `code/`，并修正 workspace root 解析；最终 smoke 与测试从归档位置重跑。上述命令记录保留实际执行时路径，不改写历史。

## 资源与边界

- 新 backbone 训练：0；checkpoint 写入：0；复用 checkpoint：2。
- 固定层：每 backbone 仅浅层 + 唯一中间层；矩阵每 backbone 10 行；logistic regression 固定 L2/lbfgs 与 C={0.1,1,10}。
- source GPU 特征抽取合计约 DF 1.85 s、VarCNNDirection 2.79 s（各角色之和）；future 五日期特征抽取合计约 DF 6.41 s、VarCNNDirection 11.15 s。线性候选拟合和读出逐项记录在 source JSON/CSV；future readout/表示字节逐日期记录在 future JSON/CSV。
- 未使用外部数据、额外层、attention、新损失、DNNF/TFAN 或 future 调参。donor checkpoints、数据和旧正式结果均只读。
