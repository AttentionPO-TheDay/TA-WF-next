# 20261003T101306Z_relative_time_structure_6622f7ed

## 问题

当前 wide+BN 生成器只使用 TAM 计数，`signed_timestamps` 仅做形状检查。本轮测试一个独立的时间结构模块：把每条流的有符号相对时间戳按固定 120 个位置区间聚合为四个无标签统计，再投影为时间 token 的残差。它是否能提高源期 valid 泛化？

## 条件与公平性

- `ce_control`：与候选完全相同的模型和 `Linear(4,128)` 参数，但时间结构输入固定为零；用于隔离新增输入与训练配方影响。
- `relative_time`：使用四个预声明特征：截断有符号均值、有效时间戳比例、截断绝对值的 log 均值、相邻有效符号切换率。
- 两条件同 seed、初始化、source index/mask stream、AdamW、12800 steps 和 valid 选模规则；三 seed 21729/23407/22026。
- 历史单视图 CE 只读参考，不访问未来日期。

## 数据与权限

TemporalDrift source15300、valid510、102类。source 标签仅用于 CE；valid 只用于预声明 checkpoint 选择和评价。相对时间特征只从每个样本输入生成，不使用标签、日期或 query 反馈。未来日期、WTT-Time、AWF、TTA 关闭。

## 指标与判定

主比较为 `relative_time - ce_control`。采用门槛：平均 valid accuracy 至少 +1pp、3/3 seed 为正、Macro-F1 均值不下降；同时报告相对历史单视图 CE。固定 mask 诊断不参与选模。

## 预算

GPU0，最多两路并行；单任务 7200 秒，总计 43200 秒；FP32、无 AMP/TF32、确定性，配方继承 wide+BN 基线。

## 启动修订 v2

2026-10-03 完成情况复核：SWA 已完成后，本 run supervisor 的任务枚举引用未定义变量 s，导致启动前退出；无 worker、参数更新或评分。保留 v1 冻结清单与 supervisor 到 artifacts，v2 仅修正 seed 枚举并补入 prepared.pt 的祖先冻结 hash；方法、配置、预算、选模规则保持原版本。SWA 反馈未用于修改本方法。

## 复核修订 v3

初版 `postrun_audit.py` 将含新增时间投影层的初始化 state 与历史 CE 的全部 key 强制相等，导致审计脚本断言失败；训练产物未被修改。复核显示 89 个共有参数与历史 CE 完全一致，四个新增时间层参数在 `ce_control`/`relative_time` 间完全一致。审计改为核对共有参数及两个新条件的新增参数配对，其他检查保持不变。
