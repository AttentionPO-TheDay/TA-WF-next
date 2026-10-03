# wide+BN 全局统计支路实验 v1

## 问题

当前单模型 wide+BN Transformer 的 valid accuracy 为 89.150%，而同协议 RF 对照为 91.699%。本轮检验当前局部生成器是否缺少稳定的全局 TAM 统计信息，并比较固定统计残差与可学习门控残差。

## 条件

- `baseline`：上一轮 wide+BN 三 seed 历史复用，仅用于公平基线和初始化审计。
- `zero_add`：新增统计支路接收全零统计量，固定加性残差；控制新增参数和训练预算的影响。
- `stats_add`：真实统计量经过 source-only 标准化、两层投影后，以加性分类 logits 残差加入基础 Transformer 输出。
- `zero_gate`：零统计量输入和可学习门控残差控制。
- `stats_gate`：真实统计量与样本表示共同生成逐类门控，再加入统计分类残差。

统计量只从当前 TAM 计算，不使用时间戳、标签或 valid 数据：4 个分段尺度（1/3/6/12）分别计算两方向均值、标准差、最大值和非零率，再计算全局四项，共 180 维。均值和标准差只由 clean source 的输入估计，训练时模型看到同一 span-mask 后的 TAM；valid 不更新统计量。

所有新条件使用与 wide+BN 相同的 3 个训练 seed、batch、12800 步、20 次 valid checkpoint 选择、AdamW、span-mask、输入和 Transformer。统计分类头初始为零，因此初始前向与 wide+BN 基线完全一致。新条件共 12 项 GPU0 训练，最多 3 路并行。

## 判定

主比较为 `stats_add−zero_add`、`stats_gate−zero_gate` 和各自相对历史 wide+BN。预定候选门槛：平均 accuracy 至少 +1 个百分点、3/3 seed 正、Macro-F1 不下降。90% 单独报告。固定三 seed 多数投票作为每个条件的附加诊断，不用 valid 搜索集成权重。

## 权限与限制

仅使用 TemporalDrift source/valid；source 标签用于梯度和指标，valid 标签只用于预定 checkpoint 选择与评价。未来日期、WTT-Time、AWF、TTA、RF teacher 和外部数据均关闭。valid 已用于多轮开发，结果是开发证据，不是独立确认。
