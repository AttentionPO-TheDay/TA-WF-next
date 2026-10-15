# 生成器局部表示选择性适应验证 v1

## 问题

在相同可训练参数、优化器、目标参考样本、批量大小和 10 次更新预算下，依据当前生成器局部表示相对源域参考统计的漂移分数，只对一半时间 token 应用残差适配器，能否比对全部 token 应用同一适配器取得更高 Macro-F1，并减少由原本预测正确变为错误的负适应？

本轮是 TemporalDrift 已观察数据上的机制筛查。它检验生成器局部表示能否提供有用的选择信号，不作为完整 Proteus 复现，也不形成论文创新性结论。

## 模型与适应位置

显式复用当前项目 `wide_bn` 三个 source-valid 最优 checkpoint（训练 seed 21729、23407、22026）。原模型、BatchNorm running statistics、Transformer 和分类器全部冻结。新增一个作用于生成器输出 `120×110` 时间 token 的零初始化瓶颈残差适配器：LayerNorm(110)-Linear(110,16)-GELU-Linear(16,110)，只有适配器参数更新。

四个条件：

1. `frozen`：原模型，无更新。
2. `global`：全部 120 个生成器时间 token 应用适配器。
3. `drift_selective`：每个样本选择相对源域逐位置均值/标准差偏移最大的 60/120 个 token。
4. `random_selective`：用与样本行号、日期、训练 seed 固定绑定的无标签随机分数选择 60/120 个 token。

后三个自适应条件共享完全相同的适配器初始化、10 个 AdamW 更新、lr=0.001、weight_decay=0、batch=64。目标函数只使用适应池预测熵。选择性条件覆盖率均固定 50%；目标标签不进入选择、梯度、停止或超参数选择。

## 数据与隔离

使用 TemporalDrift Day14、Day90、Day270。每个日期从前 5000 个 float32 有符号时间戳计算精确 SHA-256：

- 排除与既有 source/valid 精确输入相同的目标行；
- 日期内同一精确输入只保留最小原始行号；
- 对剩余唯一输入用固定、无标签 SHA-256 排序，前 1024 条作 adaptation reference，其余作独立 query；
- adaptation 与 query 的输入及重复组严格不交叉；query 输入和标签均不进入适应。

源域漂移参考统计固定使用 source 中按输入 SHA-256 排序的前 2048 条，不使用标签。原始共享数据只读；重用的 prepared tensor 与 checkpoint 只读。

## 指标与裁决

主指标为 query 上固定 102 类 Macro-F1，辅指标为 accuracy、负适应率（frozen 正确而适应后错误，占 frozen 正确样本比例）、纠错率（frozen 错误而适应后正确，占 frozen 错误样本比例）、支持集适应前后预测熵。

预注册的可行性门槛同时要求：

- `drift_selective - global` 的 Macro-F1 在 9 个 seed×日期单元平均为正，且至少 6/9 单元为正；
- `drift_selective` 的平均负适应率低于 `global`；
- `drift_selective - random_selective` 的平均 Macro-F1 为正。

不按目标标签选择 step、日期、seed、阈值或 checkpoint。全部单元和失败均报告。

## 预算与停止条件

GPU1 单进程；最多 3 小时；最多 27 个适应任务（3 seed×3 日期×3自适应条件），每项 10 个更新；不重新训练 backbone。出现 checkpoint/输入 hash 不一致、query-reference 重叠、非有限损失、模型冻结参数变化、选择覆盖率错误、CUDA OOM 或超时立即失败并保留产物。WTT-Time、AWF、其他漂移因素和外部未来测试保持关闭。
