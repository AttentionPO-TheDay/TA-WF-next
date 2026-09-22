# 实验结果

完成时间：2026-09-17（Asia/Shanghai）。本实验使用已观察的 TemporalDrift 开发数据，不是外部确认。DF 与 VarCNNDirection checkpoint 全程冻结；新增 backbone 训练/微调为 0。

## 完整性与信息权限

- 严格复用 `exp_b471517a3e6f41e7` 的 Day14/Day90/Day270、3-shot/10-shot、seeds 1729/6238/20260916 support/query。正式 3-shot 未改写；120 个 3-of-10 组合在评分前按 `resampling_manifest_v1.json` 冻结。
- DF cache 未重提取。恢复后核验现有 VarCNNDirection cache：source `(16309,512)`，Day14 `(22603,512)`，Day90 `(28599,512)`，Day270 `(19935,512)`；全为有限、逐行 L2 归一化的 `float32`，文件哈希与 cache metadata 一致。
- 先生成 18 个 frozen unit，再写入 `firewall_seal_v1.json`；seal 覆盖 34 个文件。评分时逐文件复核 seal 哈希后才读取 query 标签，并核对正式 3-shot/common-query 与 10-shot prediction 和 donor 完全一致。
- `integrity_check.json` passed：18 个 frozen unit；2,160 行重采样结果；220,320 行 site 统计；220,320 行 support-only 指标；72,104 行 repaired/harmed 归因；无临时文件。

## 预注册裁决

机器裁决为 **`B_historical_coverage_dominant`**：A 条件未通过，B 条件通过。

- **估计噪声不足以解释 3→10 gap。** 18 个 backbone×date×seed 单元的 macro-F1 gap 平均为 9.139 pp，范围 4.067–13.314 pp；120 个 3-shot 子集的单元内 SD 平均仅 1.016 pp，q90–q10 spread 平均 2.627 pp。`P(F1_subset >= F1_10-0.02)` 在 18/18 单元均为 0；q95 oracle 对 gap 的解释率在 18/18 单元均低于 50%，均值 17.8%，最大 34.7%。因此不是“大量不同 3-shot 子集可接近 10-shot”的情形。
- **新增证据主要补足 source 已知区域，但 current-new 也不可忽略。** 12,852 条新增 support 分类为：离原 3-shot 近 2,424（18.9%）、离原 3-shot 远但有同类 source 支持 5,682（44.2%）、离原 3-shot 远且 source 不支持 4,746（36.9%）。只看两类远区，54.5% 为历史 source 已支持，45.5% 为 current-new。
- **修复样本与新增近邻证据稳定关联。** 共 53,857 个 repaired、18,247 个 harmed query 事件。18/18 单元中，repaired query 获得更近新增 support 的比例都高于全 query 基线，富集为 14.0–19.8 pp。repaired query 的最近新增证据分类合计为 near-original3 9,008、far3-source-supported 32,083、far3-source-unsupported 12,766；在两类远区中，71.5% 来自 source 已支持区域。B linkage 条件为 18/18。
- **跨 seed/backbone/date 重复。** 每个 backbone×date 中，至少两个 seed 重复出现 source-supported 缺失的站点数为 85–102；current-new 为 86–101。两个 backbone 在 Day14/Day90/Day270 三个日期都满足预注册的历史覆盖重复条件。

## 不看 Query 的可识别性

预先冻结的四个 source/support-only 指标都与 site accuracy 稳定相关。最强且最一致的 `center_to_source` 风险在六个 backbone×date 组的 Spearman ρ 为 -0.730 至 -0.506，bottom-quartile failure AUC 为 0.764–0.873；其余 source coverage、support dispersion、jackknife center instability 也均达到预注册识别条件。机器裁决 `B_identifiable=true`，说明不需要 query 真标签才可发现不具代表性的 3-shot/site 风险。

这不等于已经得到可部署选择器：当前结果只证明固定指标具有事后预测关系，没有定义新的选择阈值、路由规则或更新方法。

## 简单停止条件

- 固定历史多原型 probe 没有解释主要 gap：18/18 单元均未达到 75% gap explained，范围为 -24.9% 至 63.3%，`multiprototype_stop_condition=false`。
- 既有简单 shrinkage 证据来自 `exp_a2ffb5623ad346c7`：主要表现为欠适应，未保留晚期恢复；本实验未重新调参或扩展 shrinkage。
- 因此，本实验没有触发“简单 shrinkage/固定多原型已足够解释、应停止包装新模块”的停止条件；也没有触发“只能看 query 标签才知道保留哪些区域”的不可部署停止条件。

## 最终回答

1. **3-shot 差距更像覆盖不足，而不是有限样本估计噪声。** 预注册结果为 B，且两个 backbone、三个日期、三个 support seed 一致。
2. **缺失区域以历史已有为主，但不是纯历史问题。** source-supported 区域在所有新增远区中略占多数，在 repaired query 对应的远区证据中占 71.5%；同时 current-new 占全部新增 support 的 36.9%，不能忽略。
3. **source+support 可以在不看 query 标签时识别风险。** 四个冻结指标均提供稳定预测信号，`center_to_source` 最强；但本实验没有把该信号变成新方法。
4. **按预注册诊断标准，证据足以把“历史模式保留与部分更新”保留为可研究候选，且简单 baseline/不可识别性未要求停止。** 是否实际进入机制设计由 Host 决定；本 executor 在此停止，不创建后续 method Job。

## 主要产物

- 输入与冻结定义：`DIAGNOSTIC_PLAN_v1.md`、`artifacts/input_manifest_v1.json`、`artifacts/resampling_manifest_v1.json`
- Firewall：`artifacts/firewall_seal_v1.json`
- 汇总：`artifacts/summary.json`、`artifacts/error_attribution_summary.json`、`artifacts/probe_results.json`
- 明细：`artifacts/resampling_results.csv`、`artifacts/per_site_stats.csv`、`artifacts/support_only_metrics.csv`、`artifacts/error_attribution.csv`、`artifacts/probe_results.csv`
- 独立检查：`code/verify_diagnostic.py`、`artifacts/integrity_check.json`

## 限制

- TemporalDrift 已用于方法开发；本结论不是 WTT-Time/AWF 外部确认。
- 两个 backbone 使用同一历史训练 seed 6238 的冻结 checkpoint；support seed 重复不能替代 backbone training randomness。
- “source-supported/current-new”由冻结 embedding 和 source q95 最近邻阈值定义，是可观测几何证据，不是潜在真实模式的唯一解释。
