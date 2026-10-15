# 来源

- 模型与 Proteus 损失实现为本 run 内复制的最小闭包，来自 `runs/20261010T072743Z_proteus_loss_ablation_a37b4fe6`，不通过旧工作区导入。
- checkpoint 与 prepared 数据只读引用 `runs/20261005T233151Z_wide_bn_day14_day90_eval_94f6ef85`、`runs/20261005T233927Z_wide_bn_later_dates_eval_5add2406`。
- split、历史损失消融输入和完整配方仅用于复用/核对，主比较在本 run 内重新产生 `full_control`。
- 新增代码实现固定候选预算下的无标签区间接受/回退与独立审计。

- v2 仅修复回退快照跨 epoch 的实现错误；v1 失败产物保留在 `runs/20261010T145409Z_proteus_rollback_budget800_2b9fa69a`。
