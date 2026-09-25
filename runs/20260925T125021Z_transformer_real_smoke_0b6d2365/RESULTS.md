# 实验结果

状态：completed；source-only 真实 trace 结构 smoke 通过。

从 TemporalDrift `train.npz` 复用固定 source 行 `[8800, 15843, 687, 10601]`，只生成 packet/run/window token，不读取 valid 或未来日期。四条 trace 的 observed packet 数为 1336、1261、780、1075；token shape 为 packet `[4,100,2]`、run `[4,128,4]`、window `[4,120,4]`。

单步结果：无标签 span 重建 loss 0.45775（遮挡 token 50 个）；有标签 CE loss 4.36531（仅证明梯度路径存在）；adapter-only TTA loss 0.0000225，teacher 置信度阈值 0.8 下选中 0 条，双视角 consistency 0.000225。TTA 后 teacher 未改变、student 非 adapter 参数未改变、adapter 参数已改变。

这不是性能实验：没有 checkpoint、没有 valid/F1、没有 epoch 选择、没有未来数据评分。TTA 置信度筛选为 0 也符合保守原型预期，不能解释为方法失败或成功。原始数据只读。
