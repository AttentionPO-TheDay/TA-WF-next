# 生成器固定统计与学习特征分离适应验证 v1

## 问题

将生成器输出的 30 维固定计数统计与 80 维学习局部特征分离后，比较完整 110 维适应和只修改 80 维学习特征的适应，检验特征分离能否提高 Macro-F1 并减少负适应。学习特征的漂移分数再与随机 token 选择比较。

本轮是 TemporalDrift 已观察数据上的机制筛查。它检验生成器局部表示能否提供有用的选择信号，不作为完整 Proteus 复现，也不形成论文创新性结论。

## 模型与适应位置

显式复用当前项目 `wide_bn` 三个 source-valid 最优 checkpoint（训练 seed 21729、23407、22026）。原模型、BatchNorm running statistics、Transformer 和分类器全部冻结。新增零初始化瓶颈残差适配器：完整条件作用于 `120×110`，学习条件只作用于 `120×80` 的学习特征；固定 30 维计数统计直接保留。

四个条件：

1. `frozen`：原模型，无更新。
2. `full_global`：110 维全部时间 token 应用适配器。
3. `full_masked_learned_global`：保留与 `full_global` 相同的 110 维适配器结构和初始化，但把前 30 维固定统计的适配残差屏蔽为零。
4. `learned_global`：只对 80 维学习特征的全部时间 token 应用适配器。
5. `learned_drift_selective`：只对 80 维学习特征，并选择相对源域统计偏移最大的 60/120 个 token。
6. `learned_random_selective`：只对 80 维学习特征，用固定无标签随机分数选择 60/120 个 token。

后三个学习特征条件共享完全相同的适配器初始化、10 个 AdamW 更新、lr=0.001、weight_decay=0、batch=64；两个完整条件使用相同规则初始化的 110 维适配器。目标函数只使用适应池预测熵。选择性条件覆盖率均固定 50%；目标标签不进入选择、梯度、停止或超参数选择。

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

- `learned_global - full_global` 的 Macro-F1 平均为正；
- `learned_drift_selective - learned_global` 的 Macro-F1 平均为正，且至少 6/9 单元为正；
- `learned_drift_selective` 的平均负适应率低于 `learned_global`；
- `learned_drift_selective - learned_random_selective` 的 Macro-F1 平均为正。

不按目标标签选择 step、日期、seed、阈值或 checkpoint。全部单元和失败均报告。

## 预算与停止条件

GPU1 单进程；最多 3 小时；最多 27 个适应任务（3 seed×3 日期×3自适应条件），每项 10 个更新；不重新训练 backbone。出现 checkpoint/输入 hash 不一致、query-reference 重叠、非有限损失、模型冻结参数变化、选择覆盖率错误、CUDA OOM 或超时立即失败并保留产物。WTT-Time、AWF、其他漂移因素和外部未来测试保持关闭。
