# Execution record

日期：2026-09-16（Asia/Shanghai）；工作目录始终为 `/home/rbf/TA-WF-next`。

- 在读取 donor Day14/90/270 query 指标前完整读取项目协议与 donor 冻结计划，核对 checkpoint、split、manifest、隔离审计和 G-source 哈希；随后冻结 `SUPPORT_SELECTION_PLAN.md`，SHA-256 `89c3059fefc78fe47151e8a25e8107f64171542597d87af335539e401e3791dc`。
- preflight 在计划冻结后完整解析 donor/history 的 results、summary、metrics、integrity、20.7 MB manifest、36 个 evaluation 哈希、split 与 checkpoint history；0 errors。
- CPU 冻结提取 source prototype 和两个 backbone×三个日期的 embedding；每配置只在 support 内执行预注册的 3/5-fold CV，再用完整 support 重拟合。一个 backbone×date 的 6 个配置全部预测固定后才读取 common-query truth。
- 验证命令类别：`py_compile`；实验单元测试（3 passed）；`preflight`；`prepare-source`；6 次 `evaluate`；`summarize`；`analyze_results.py`；独立 `verify_support_selection.py`。
- 首次 `prepare-source` 会话在工具返回边界附近留下 DF prototype；恢复时将其移到 `logs/df_source_prototypes.interrupted.npy`，正式重跑后发现原会话已异步完成 VarCNNDirection 与 `source_preparation.json`。正式 source preparation 记录和两个正式 prototype 均完整，recovery 副本未被引用。
- 新 backbone 训练/微调：0；checkpoint 输出：0；外部数据/TTA/adapter/attention/memory/DNNF/TFAN/多日期连续适应：0。
