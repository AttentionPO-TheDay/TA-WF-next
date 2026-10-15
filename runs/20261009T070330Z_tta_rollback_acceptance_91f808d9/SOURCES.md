# 来源

- 模型结构最小闭包复制自本项目 run `20261009T053533Z_generator_factor_split_adaptation_87d59d18`；该实现最终来源于当前项目的 wide+BN 模型，不通过 `sys.path` 或隐式导入加载旧工作区代码。
- 三个 wide+BN checkpoint、source/valid、Day14、Day90 prepared tensor 只读引用 run `20261005T233151Z_wide_bn_day14_day90_eval_94f6ef85`；Day270 prepared tensor只读引用 `20261005T233927Z_wide_bn_later_dates_eval_5add2406`。
- split 和全局适应预测只读对照 run `20261009T053533Z_generator_factor_split_adaptation_87d59d18`，用于验证本轮基线精确复现。
- 新增代码仅实现无标签逐步接受、整体验证回退、结果汇总和独立复核。共享数据保持只读。
