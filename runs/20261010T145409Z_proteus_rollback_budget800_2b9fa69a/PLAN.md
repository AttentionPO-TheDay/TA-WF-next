# Proteus 无标签接受与回退（匹配 800 次候选更新预算）

问题：完整 Proteus 在 Day14 退化、Day90/270 获益且负适应率较高。本实验检验，在每个条件均尝试相同 800 次候选更新的情况下，用 support 上可见的无标签信号接受或回退更新，能否保留远期收益并减少负适应。

## 条件与匹配预算

四个新条件均使用当前 wide+BN 三个 checkpoint、同一 source/target/pseudo 抽样流、相同四项 Proteus 损失、Adam lr=0.001、100 epoch×8 batch=800 次候选更新：

- `full_control`：全部候选更新接受；
- `rollback_10`：每 10 次候选更新检查一次；
- `rollback_25`：每 25 次候选更新检查一次；
- `rollback_50`：每 50 次候选更新检查一次。

回退条件只使用完整 support 的平均预测熵和高置信度比例：相对上一次接受状态，熵至少下降 1e-7，且高置信度比例不下降超过 1e-7。拒绝时恢复该检查区间开始的模型和 Adam 状态。GMM 每 5 epoch 重新拟合；回退不改变已生成的抽样流，下一次 GMM 刷新会基于当前模型重新计算。

## 权限与数据

仅 TemporalDrift Day14/90/270（已观察开发数据）。source 标签仅用于 source CE；support 适应对象只含 timestamps、TAM、rows，support 标签不进入适应；query 输入只在适应结束后推理，query 标签只用于固定预测后的指标。支持集和 query 精确复用上一轮 split，外部未来、WTT-Time、AWF 及其他因素关闭。

## 指标与裁决

主指标为固定 102 类 Macro-F1；同时记录 accuracy、负适应率、纠错率、改变率、接受/拒绝次数、support 熵与置信度轨迹、参数和 BN 变化。每个 rollback 条件相对同轮 `full_control` 比较。预注册筛查信号为 Macro-F1 平均提高至少 0.5pp、负适应率下降至少 1pp、且至少 6/9 seed×日期单元 F1 提高；所有日期和失败结果完整报告。该信号仅用于 TemporalDrift 开发筛查，不代表统计显著性。

## 审计与资源

保存每项 checkpoint、query logits、损失/GMM轨迹、接受轨迹、输入流哈希和配置冻结哈希。独立审计重算指标、验证 support/query 隔离、source/target 抽样流跨条件一致、重载 36 个 checkpoint 并核对预测。GPU0 单进程，单任务最多 2400 秒，整批最多 21600 秒；OOM、非有限值、split/hash 不一致或审计失败即停止并保留产物。
