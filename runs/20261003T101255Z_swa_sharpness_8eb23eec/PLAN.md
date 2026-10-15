# 20261003T101255Z_swa_sharpness_8eb23eec

## 问题

固定 wide+BN 单视图 CE 训练轨迹，在后段学习率已降到 0.0001 后对七个预注册 checkpoint 做随机权重平均（SWA），能否提高 valid 泛化？同一训练轨迹的 CE 最佳 checkpoint 作为严格控制。SWA 不改变输入、损失、优化器或训练梯度。

## 条件与公平性

- `ce_final`：单条 CE 轨迹中按预注册 valid 选模规则得到的最佳 checkpoint。
- `swa`：同一条 CE 轨迹在 9920、10400、10880、11360、11840、12320、12800 步的浮点 state tensor 等权平均；整数 BN 计数 buffer 取最后快照。窗口和权重在训练前冻结，不根据 valid 调整，不做 BN 重校准。
- 三个 seed 21729/23407/22026 共三条轨迹；固定 source index stream、mask stream、初始化和 12800 optimizer steps。
- 历史 CE 仅作只读参考，不从历史 checkpoint 生成 SWA。

## 数据与权限

TemporalDrift source15300、valid510、102类；source 标签仅用于 CE，valid 仅用于预声明 checkpoint 选择和评价。未来日期、WTT-Time、AWF、TTA 全部关闭。

## 指标与判定

主比较为 `swa - ce_final`。采用门槛：平均 valid accuracy 至少 +1pp、3/3 seed 为正、Macro-F1 均值不下降；同时报告与历史单视图 CE 的差异。固定 mask 诊断不参与选模。

## 预算

GPU0，最多两路并行；单任务 7200 秒，总计 43200 秒；FP32、无 AMP/TF32、确定性，训练配方继承 wide+BN 基线。
