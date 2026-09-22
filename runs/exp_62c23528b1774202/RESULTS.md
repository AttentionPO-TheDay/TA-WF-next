# 实验结果

完成时间：2026-09-17（Asia/Shanghai）。本轮只使用 TemporalDrift、冻结 DF/VarCNNDirection、正式每类 3-shot 和 source-only 冻结历史库；新增 backbone 训练/微调、checkpoint、表示更新、TTA、外部数据和超参搜索均为 0。

## 直接结论

**否。相同历史存储预算下，区域级历史冲突抑制 D 没有比普通多原型 B 或网站级整体混合 C 稳定更有效，也没有超过现有最强 support-CV 3-shot A。** `initial_mechanism_signal=false`；预冻结的复杂区域模块停止条件触发。该机器裁决不替 Host 作后续研究路线决定。

Day90/Day270 的 12 个 backbone×date×seed 单元中，D 相对 A/B/C 的 macro-F1 平均差分别为 **-2.883/-0.343/+0.382 pp**，正差单元分别为 **1/12、3/12、11/12**。相对 C 的小幅正差低于冻结的 +1.0 pp 稳定优势门槛；B 与 C 在两个 backbone 分别合并后期都落入与 D 的 `<1.0 pp` “实质相当”区间。

去区域消融没有支持机制归因：D 相对统一网站保留消融在后期为 **+0.382 pp、11/12 单元为正**，低于冻结的 `+0.5 pp 且至少 9/12` 联合门槛。网站级平滑 C 与统一二元消融在全部 408,462 条 unit-query predictions 上恰好相同；因此不能把 D 的小幅差值归因成已成立的区域粒度收益。

Day14 明显受损：D 相对 A 的 seed-mean macro-F1 在 DF 为 **-6.708 pp**，VarCNNDirection 为 **-10.464 pp**，远超允许的 1 pp 损伤。两个 backbone 后期也都未超过 A：DF -3.313 pp，VarCNNDirection -2.454 pp。

## 绝对 macro-F1

单元格为三个正式 support seed 的 mean±sample-SD，单位为百分数。

| Backbone | 日期 | A support-CV | B 普通多原型 | C 网站整体权重 | D 区域抑制 | D 去区域 |
|---|---|---:|---:|---:|---:|---:|
| DF | Day14 | 72.393±0.077 | 66.007±0.281 | 64.434±1.096 | 65.685±0.579 | 64.434±1.096 |
| DF | Day90 | 57.249±1.376 | 54.442±0.626 | 54.329±0.967 | 54.657±1.060 | 54.329±0.967 |
| DF | Day270 | 49.318±0.310 | 45.964±0.193 | 45.033±0.194 | 45.285±0.154 | 45.033±0.194 |
| VarCNNDirection | Day14 | 70.571±0.228 | 60.537±0.883 | 59.227±0.972 | 60.107±1.015 | 59.227±0.972 |
| VarCNNDirection | Day90 | 56.262±1.035 | 53.062±0.492 | 52.117±0.629 | 52.801±0.765 | 52.117±0.629 |
| VarCNNDirection | Day270 | 49.040±2.189 | 48.240±0.663 | 47.329±1.153 | 47.594±0.932 | 47.329±1.153 |

10-shot 仅保留为前序高标签预算背景，没有进入本轮任何规则、预测或公平胜负。

## 预算、防火墙与完整性

- B/C/D/去区域逐字节读取同一个每-backbone历史包：306 个 512 维 float32 exemplar、306 个 int64 source row id、102 个 float32 source q95 阈值；数值 payload **629,544 bytes**，实际 NPZ **630,326 bytes**。每个方法都物化相同 `(102,3)` float32 运行态权重数组（1,224 bytes）。D 没有额外历史条目或历史信息。
- 18 个配置共抑制 361 个 region-instance。正式输入均为 306 条 3-shot support；额外七条未用于算法。共同 query 只为保持与已冻结 A 完全一致，沿用前序预定的 full-10 exclusion。
- 在评分前生成 `firewall_seal_v1.json`，覆盖计划、配置、代码、输入 manifest、两个共享历史包和 18 个未评分预测。评分后独立重算 90 行指标、2,042,310 条 method prediction；`integrity_check.json` 为 `passed=true`、0 errors。
- 普通多原型 B 的逐条预测与 `exp_04faf4088b604155` 既有冻结 probe 完全一致；A 的逐条预测直接引用 `exp_376fca9354214097` 正式 `support_selected_baseline`，未重选弱 donor。

## 产物

- 冻结方法：`METHOD_PLAN_v1.md`；依赖哈希：`artifacts/input_manifest_v1.json`；预测 seal：`artifacts/firewall_seal_v1.json`
- machine-readable：`artifacts/metrics_long.csv`、`artifacts/deltas.csv`、`artifacts/summary.json`、`artifacts/ablation_results.json`
- 预算与完整性：`artifacts/budget_audit.json`、`artifacts/integrity_check.json`
- 执行记录：`execution_record.md`

本实验至此停止；未创建或启动后续实验。
