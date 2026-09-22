# 有界既往暴露检索

2026-09-22，检索旧 `/home/rbf/TA-WF/{configs,experiments,docs,outputs}` 下 md/yaml/*config*.json/*summary*.json/csv：
`VersionDrift|NetworkDrift|BehaviorDrift|version_drift|network_drift|behavior_drift`，不区分大小写，含 ignored 文件。

命中仅官方README快照及 exp_d839f7ce217f488a 六场景审计。新工作区排除本轮两次审计后，runs 的 PLAN.md/RESULTS.md/config.json 未命中上述路径名。补充文件路径名检索命中既有审计和无关network数据获取记录。

结论：本次有界检索未发现三组已训练/适应/性能选择的直接记录，不是全盘未使用证明。别名、聚合配置、日志、外部工作区或未记录运行不在证明范围内。旧审计已看过标签/抽样X；本次全量X/y内容匹配和Behavior URL审计新增数据暴露。它们不可称研究者从未见过的数据；方法选择独立性需在最终确认前另核。
