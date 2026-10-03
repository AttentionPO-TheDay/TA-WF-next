# wide+BN 集成审计

## 问题

固定现有 wide+BN 逐级生成器＋Transformer 的三个训练 seed，检查 seed、分类头和官方 RF 强对照的错误互补性，判断 89.150% 到 90% 的差距能否通过预注册的输出融合获得。

## 数据与权限

- 数据仅使用上一轮冻结的 TemporalDrift `source` 15300 条和 `valid` 510 条预测输出。
- `source` 只用于报告训练集表现和错误互补诊断；不重新训练、不拟合融合权重。
- `valid` 只用于固定规则下的评价，不用于选择融合权重。
- TemporalDrift future、WTT-Time、AWF、TTA 和漂移适应均关闭。

## 固定输入与方法

预测文件来自已完成并审计的 wide+BN 三 seed、progressive MLP 三 seed 和 matched RF 三 seed。所有融合权重预先固定，不根据 valid 结果调节：

1. 单个 wide+BN seed；
2. 三个 wide+BN 的 raw-logit 平均；
3. 三个 wide+BN 的逐样本 class-wise z-score 后平均；
4. 三个 wide+BN 的多数投票，平票由 z-score logits 和决定；
5. wide+BN z-score 集成与 progressive MLP z-score 集成按 75/25、50/50、25/75 融合；
6. wide+BN z-score 集成与 matched RF z-score 集成按 75/25、50/50、25/75 融合。RF 结果只作强对照融合诊断，不称为自有生成器模型。

逐样本 z-score 只改变类别分数尺度，不使用标签。主门槛为 wide+BN seed 集成相对单 seed 平均是否提高，并报告是否达到 90%；RF 融合单列，不作为自有模型准入。

## 预算与停止条件

无训练、无 GPU、固定 CPU 分析；只读取已有 logits 和标签，保存每种方法的 source/valid 预测及指标。若输入行数、类别数或预测 hash 不一致则停止。不得新增融合权重搜索。

