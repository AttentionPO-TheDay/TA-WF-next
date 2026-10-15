# 无标签更新接受与回退实验 v1

## 问题

当前完整 110 维全局熵最小化适应相对冻结模型的 Macro-F1 仅提高约 0.116 个百分点，同时产生约 1.032% 的负适应率。本实验检验：在相同的 10 次无标签更新尝试预算下，使用 support 上可观测的预测熵和高置信度比例决定是否接受更新，能否保留全局适应的收益并减少负适应。

本轮只使用 TemporalDrift 已观察开发数据。它是安全适应机制筛查，不作为完整 Proteus 复现，也不访问外部未来测试。

## 模型、条件与更新规则

复用 `wide_bn` 三个 source-valid 最优 checkpoint（训练 seed 21729、23407、22026）。生成器、BatchNorm、Transformer 和分类器全部冻结；只训练时间 token 后的零初始化 `110→16→110` 瓶颈残差适配器。

四个条件共享相同 adapter 结构、初始化、AdamW、lr=0.001、weight_decay=0、batch=64 和确定性 minibatch 顺序：

1. `frozen`：不适应。
2. `global`：连续执行 10 次预测熵最小化更新，不回退。
3. `stepwise_rollback`：每次更新后在完整无标签 support 上计算平均预测熵和最大类别概率不低于 0.8 的样本比例。只有熵相对上一接受状态至少下降 `1e-7`，且高置信度比例不低于上一接受状态（容差 `1e-7`）时接受；否则同时恢复 adapter 和 optimizer 到更新前状态。始终尝试 10 次。
4. `episode_rollback`：连续执行与 `global` 相同的 10 次更新；若最终完整 support 熵未比初始状态至少下降 `1e-7`，则恢复初始 adapter。该条件的回退判定只使用熵。

所有接受和回退判定均不使用标签。support 额外前向只作为无标签监控，不产生额外梯度更新。

## 数据与权限

使用 TemporalDrift Day14、Day90、Day270，并精确复用 run `20261009T053533Z_generator_factor_split_adaptation_87d59d18` 的 split：每个日期 1024 条无标签 support，其余唯一且与 source 不重叠的样本作为独立 query。运行时重新计算 split 并逐行断言一致。

- source 输入：仅用于复用冻结 checkpoint；不使用 source 标签或重新训练。
- target support 输入：用于熵最小化、无标签接受判定和回退；support 标签完全不可用。
- query 输入：只在适应结束后推理。
- query 标签：预测固定后计算指标；不得用于更新接受、步数、阈值、超参数或条件选择。
- WTT-Time、AWF、Version/Network/Behavior 和其他外部未来数据保持关闭。

## 指标与裁决

主指标是 query 上固定 102 类 Macro-F1。辅指标是 accuracy、负适应率、纠错率、预测改变率、接受更新数、support 熵轨迹、高置信度比例轨迹和特征相对改变量。

`stepwise_rollback` 是预注册主方法，必须同时满足：

- 平均 Macro-F1 不低于 `global`；
- 平均负适应率相对 `global` 至少降低 0.2 个百分点；
- 9 个 seed×日期单元中至少 6 个单元的负适应率低于 `global`。

`episode_rollback` 作为预注册次要方法按同一门槛单独报告，不用其结果替换主裁决。所有日期、seed 和失败结果完整报告。

## 预算与停止条件

单 GPU 进程；3 seed×3 日期×3 个适应条件，共 27 个适应任务；每项最多 10 次梯度更新；最多 3 小时。不重新训练 backbone。输入或 checkpoint hash 不一致、split 不一致、query/support 重叠、非有限损失、冻结参数变化、旧 global 预测无法精确复现、CUDA OOM 或超时均立即失败并保留产物。
