# 来源

本 run 验证生成器输出的固定 30 维计数统计与 80 维学习特征是否应在无标签适应时分开处理。新增 `full_masked_learned_global` 是完整适配器的同结构、同初始化输出屏蔽对照。

- 模型结构最小闭包复制自当前项目 `runs/20261005T233151Z_wide_bn_day14_day90_eval_94f6ef85`，该 run 又冻结自 `20261002T053706Z_progressive_capacity_bn_mixup_1ad233da`。本 run 不通过 `sys.path` 加载旧工作区代码。
- 三个 wide+BN checkpoint、source/valid prepared tensor、Day14/Day90 prepared tensor只读引用前述 run；Day270 prepared tensor只读引用 `runs/20261005T233927Z_wide_bn_later_dates_eval_5add2406`。
- 新增代码仅为本 run 的局部适配器、无标签选择规则、隔离 split、执行与复核代码。
- 原始数据位置由 `configs/datasets.json` 登记；本轮不改写共享数据。
