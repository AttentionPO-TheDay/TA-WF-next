# 实验结果

状态：completed。CPU 4 threads，3 seeds（1729/3407/2026），15 epochs；GPU、WTT-Time、AWF 均未访问。抽样复用上一轮封存 manifest，未重新随机划分。

## 结果（macro-F1，百分比；三 seed 均值）

| 条件 | valid | Day14 | Day30 | Day90 | Day150 | Day270 |
|---|---:|---:|---:|---:|---:|---:|
| window_full | 25.363 | 23.822 | 22.836 | 18.845 | 16.398 | 13.934 |
| window_adapter | 24.084 | 23.240 | 21.131 | 18.430 | 15.958 | 13.877 |
| window_tail_neutral | 26.125 | 24.658 | 23.141 | 19.105 | 16.610 | 14.545 |

相对 full，adapter 的 valid 为 -1.279 pp，未来日期为 -0.582/-1.705/-0.415/-0.440/-0.057 pp，未达到“valid 不下降且多数日期正向”的门槛。三日期中 adapter 仅 Day270 的均值近似持平，不能称适配器改善泛化。tail-neutral 的 valid +0.762 pp，Day14/30/90/150/270 分别 +0.836/+0.305/+0.260/+0.212/+0.611 pp；这是诊断性结果，说明当前 window 表示中 partial/tail 的连续比例可能包含噪声或采集模式，不能直接解释为新的抗漂移方法。

## 解释与限制

适配器在本预算下没有正向证据，反而引入了 source/valid 之间更强的拟合波动；不能因为它是零初始化或参数可训练就保留。tail-neutral 的改善值得复核，但它改变了输入语义且使用了人为常量，可能只是去除了不稳定尾部特征，尚不能证明生成器本身更有效。所有未来日期均属于 TemporalDrift 方法开发评分，不是独立确认；本实验也未验证在线漂移适配。

## 完成检查

- 预测文件、指标和最佳 epoch 已写入 artifacts；每个条件×seed 均使用 valid macro-F1 选模。
- `manifest.json` 记录了固定 sampling manifest、partial window 计数及预测 SHA-256。
- source/valid/未来角色未重新抽样；原始数据只读。
