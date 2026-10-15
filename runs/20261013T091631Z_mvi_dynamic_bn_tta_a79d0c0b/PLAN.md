# TemporalDrift 闭环最小可行机制（MVI）

## 问题

在当前 wide+BN Transformer 上，Proteus 低强度（0.25 倍 Adam 学习率）已达到 Macro-F1 73.432%，但仍有约 4.05% 负适应。验证一个不使用伪标签的闭环机制：以目标预测熵控制学习率，并用同一无标签置信信号插值 BatchNorm running-stat 更新；另测跳过高不确定性更新是否降低计算开销和负适应。

## 新条件

五个条件均使用同一 source/target 抽样、3 seed、Day14/Day90/Day270、100 epoch、800 次候选更新、source CE + target entropy/diversity + feature MMD；完全移除 GMM、伪标签和伪标签前向：

- `no_pseudo_fixed`：固定 lr=0.00025，BN 默认 momentum=0.10。
- `dynamic_lr_only`：每步先 eval/no_grad 获得目标 batch 熵（不更新 BN），相对初始 support 平均熵缩放 lr，scale=clip(1/relative_entropy,0.25,1.0)，BN momentum 固定 0.10。
- `bn_interp_only`：lr 固定 0.00025，目标前向 BN momentum=0.01+0.09*scale，source 前向固定 0.10，scale 同上。
- `mvi_joint`：同时使用动态 lr 和 BN momentum 插值。
- `mvi_joint_skip`：`mvi_joint` 规则；先在 eval/no_grad 下预测 target batch，仅对归一化熵 H/log(102) 位于 [0.02,0.50] 的目标样本反传；没有保留样本时跳过整个候选更新且不更新 BN。

初始 support 熵、当前 batch 熵、lr scale、BN momentum、跳过/执行更新数全部记录。support 只有输入和行号，不传入标签；query 只在适应完成后推理评分。

## 历史参考

复用同一 query split 的 frozen、Proteus strength_025 和 rollback_10 预测，明确标记 `historical_reference`，不重复训练。strength_025 为当前最强 Proteus 基线，Macro-F1 73.432%。

## 评价与目标

主指标为 Macro-F1 和负适应率；同时报告 accuracy、纠错率、逐日期结果。目标是 aggregate Macro-F1 >74.5% 且负适应率 <2%。记录每任务 forward/backward/optimizer/task/query-inference 时间、每 Trace 推理延迟、显存峰值和执行/跳过更新数。目标是预注册筛选线，不把未达到目标的结果丢弃。

## 数据与边界

仅 TemporalDrift 已观察开发数据。WTT-Time、AWF 和 Version/Network/Behavior 不在本轮使用；TemporalDrift 的 signed timestamp 输入不能直接当作其他数据集的输入。若 MVI 在 TemporalDrift 冻结，另立跨数据集方法版本并先审计数据角色、输入模态、训练/验证/测试权限后执行。

## 资源

GPU1 单进程，45 个新任务；历史参考 27 行。每任务最长 2400 秒，批次最长 21600 秒；所有新条件共享更新预算和 loader/dropout seed 规则。

成本定义：逐更新同步记录 probe/训练前向/反向/optimizer 秒数、target 样本暴露量和实际更新数；反向单 trace 值为 batch 摊销，不称逐条在线真实延迟。support-only batch=1/128 的5次warmup+30次定时重复用于推理 p50/p95；不使用query作为计时调参输入。GPU1与既有进程共享，记录资源快照，不宣称独占硬件速度。固定条件也做相同 eval probe，便于配对消融；实际部署可取消其无用probe，故额外报告probe开销。

完全去伪标签包括移除第三路前向，因此BN累计次数也变化；与历史no_pseudo（三路前向仍保留）不是同一干预。本轮主对照为历史强基线及内部2×2消融，不重复执行已完成旧训练。跳过条件是独立次要候选，800候选机会而非保证800实际更新；完整MVI是预定主候选，不事后按最好条件冒充主候选。
