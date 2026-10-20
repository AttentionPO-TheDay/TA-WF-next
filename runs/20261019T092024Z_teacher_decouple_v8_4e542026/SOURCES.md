# 实现和历史来源

基础模型、输入、LoRA、candidate/source-neighbor、RefinedState、momentum_legacy、routing、frontier_base、isolation的loss_parts显式复制自v7 run 20261019T074232Z_mechanism_isolation_v7_50f41beb（原始链见其SOURCES.md）。本run独立导入，无旧项目隐式sys.path。runtime只改本run状态前缀。新增decoupling.py/worker.py/scoring.py/audit.py/report.py/controller.py属本轮。BN校准/写入隔离旧函数作为来源保留但本run不得调用。控制仅显式加载已核验v6 sharp_target_only，源模型仅来自纯净source_run。v7旧结果仅用于当前机制设计及评分对照，不改其封存。
