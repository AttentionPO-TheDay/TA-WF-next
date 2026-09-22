# exp_4b72ed0c9a354d54

问题：在不重训 backbone 的条件下，用有上限的 source-only 表示/读出矩阵区分浅层表示不足、顺序聚合损失与余弦近邻读出不足；只有按预注册规则得到可用配置后才冻结进入已观察 TemporalDrift 日期评分。

状态：completed。完整 source 矩阵与 `SOURCE_RESULTS.md` 已在任何本实验 future 抽取前落盘；`artifacts/frozen_selection.json` 已按 source validation 规则冻结每 backbone 唯一配置及 artifact 哈希，随后完成五个获准日期的固定评分。

数据、权限、checkpoint、层、读出、指标、门槛、预算和停止条件详见 `representation_diagnostic_plan.md` 与 `config.json`。本实验新增 backbone 训练为 0；所有可变产物仅写入本目录。
