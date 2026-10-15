# 来源

实现复用：`runs/20261012T155207Z_proteus_lowlr_loss_attribution_14931ed8` 的 run-local wide+BN/Transformer、TAM 生成器和指标工具；本实验新增 runner、audit、配置和成本记录，未通过旧项目 sys.path 导入。

基座 checkpoint：`runs/20261005T233151Z_wide_bn_day14_day90_eval_94f6ef85`。
历史参考：Proteus strength_025 来自 `runs/20261011T042912Z_proteus_strength_sweep800_b022cde2`；rollback_10 来自 `runs/20261011T041813Z_proteus_rollback_budget800_v2_4b96c033`；冻结预测来自低学习率归因 run。所有历史参考均按同一 TemporalDrift query split 复用并标记。
