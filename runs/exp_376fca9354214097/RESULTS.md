# Support-internal selection results

## 完成状态与主结论

实验已按冻结的 `SUPPORT_SELECTION_PLAN.md` 完成。新增 backbone 训练/微调、checkpoint 写入均为 0；只读复用 donor 的两个正式 checkpoint、512 维冻结 global embedding、G-source、18 个 support manifest 与 current 标签权限。3-shot/10-shot 在每个 date×seed 上均使用 canonical pool 排除完整 10-shot support 后的同一 common query；3-shot 剩余 7 条既未训练/选参，也未返回 query。

严格按预注册门槛，**10-shot support-internal 选择没有完全解决主要权衡**：两 backbone 都保留了 Day90/270 大部分 current-only recovery；VarCNNDirection 也通过 Day14 保护，但 DF Day14 的 selected baseline 等于 G-source，只比较强 current-only 高 **1.951 pp** macro-F1，距 2.0 pp 门槛差 **0.049 pp**，因此跨 backbone Day14 严格未通过。事后 query oracle 的候选 envelope 在 DF Day14 可高出较强 current-only **2.023 pp** 并完整通过全部门槛，所以主失败归因为预注册的 **A：selection failure**，不是 B（候选族必然不足）。oracle 只用于归因，不是最终方法性能。

## common-query 绝对性能

单元格为三个 support seed 的 `accuracy±SD / macro-F1±SD`，单位为百分数。`fixed α` 是 alpha=.25，`fixed λ` 是 lambda=.10；`sel-P/L/X` 分别为 support-selected prototype、linear 与预注册跨家族 baseline。

### DF

| 日期 | shot | G-source | Current proto | G-current | fixed α | fixed λ | sel-P | sel-L | sel-X |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day14 | 3 | 72.56±0.07 / 72.39±0.07 | 64.56±1.56 / 63.93±1.54 | 64.70±1.52 / 64.10±1.54 | 70.82±0.16 / 70.59±0.13 | 72.54±0.11 / 72.38±0.12 | 71.81±1.36 / 71.63±1.40 | 72.55±0.07 / 72.39±0.08 | 72.55±0.07 / 72.39±0.08 |
| Day14 | 10 | 72.56±0.07 / 72.39±0.07 | 70.12±0.12 / 69.81±0.15 | 70.67±0.13 / 70.44±0.16 | 71.13±0.09 / 70.92±0.07 | 72.56±0.06 / 72.41±0.07 | 72.56±0.07 / 72.39±0.07 | 72.56±0.07 / 72.39±0.07 | 72.56±0.07 / 72.39±0.07 |
| Day90 | 3 | 57.28±0.03 / 55.45±0.02 | 55.29±0.61 / 54.16±0.85 | 55.77±0.58 / 54.66±0.75 | 57.34±0.10 / 55.83±0.09 | 57.29±0.03 / 55.53±0.03 | 58.53±1.21 / 57.15±1.53 | 57.28±0.25 / 55.76±0.21 | 58.44±1.35 / 57.25±1.38 |
| Day90 | 10 | 57.28±0.03 / 55.45±0.02 | 62.25±0.16 / 61.54±0.14 | 63.72±0.24 / 63.25±0.21 | 57.57±0.16 / 56.09±0.15 | 57.31±0.03 / 55.55±0.02 | 62.38±0.14 / 61.59±0.10 | 63.72±0.24 / 63.25±0.21 | 63.72±0.24 / 63.25±0.21 |
| Day270 | 3 | 46.71±0.01 / 43.90±0.01 | 49.03±0.92 / 47.28±0.87 | 49.46±1.04 / 47.83±0.93 | 48.54±0.15 / 45.92±0.20 | 46.83±0.05 / 44.07±0.05 | 51.77±0.41 / 49.32±0.31 | 48.06±1.87 / 45.87±2.56 | 51.77±0.41 / 49.32±0.31 |
| Day270 | 10 | 46.71±0.01 / 43.90±0.01 | 57.25±0.01 / 55.99±0.26 | 58.97±0.34 / 58.15±0.45 | 48.99±0.10 / 46.42±0.10 | 46.78±0.02 / 44.02±0.01 | 57.25±0.01 / 55.99±0.26 | 58.97±0.34 / 58.15±0.45 | 58.97±0.34 / 58.15±0.45 |

### VarCNNDirection

| 日期 | shot | G-source | Current proto | G-current | fixed α | fixed λ | sel-P | sel-L | sel-X |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day14 | 3 | 70.89±0.05 / 70.75±0.05 | 55.23±2.79 / 54.48±2.93 | 55.07±2.73 / 54.34±2.80 | 68.48±0.14 / 68.13±0.10 | 70.90±0.12 / 70.76±0.12 | 70.89±0.05 / 70.75±0.05 | 70.72±0.23 / 70.57±0.23 | 70.72±0.23 / 70.57±0.23 |
| Day14 | 10 | 70.89±0.05 / 70.75±0.05 | 65.74±0.74 / 65.28±0.75 | 66.25±0.53 / 65.95±0.58 | 69.04±0.11 / 68.73±0.10 | 70.92±0.04 / 70.79±0.03 | 70.89±0.05 / 70.75±0.05 | 70.94±0.07 / 70.83±0.07 | 70.94±0.07 / 70.83±0.07 |
| Day90 | 3 | 57.32±0.07 / 55.47±0.05 | 51.30±0.23 / 50.41±0.26 | 52.09±0.32 / 51.17±0.36 | 56.64±0.15 / 55.21±0.14 | 57.31±0.04 / 55.53±0.05 | 57.67±0.58 / 56.12±1.12 | 57.24±0.03 / 55.85±0.40 | 57.60±0.64 / 56.26±1.04 |
| Day90 | 10 | 57.32±0.07 / 55.47±0.05 | 61.43±0.51 / 60.87±0.62 | 63.21±0.34 / 62.79±0.42 | 57.10±0.12 / 55.69±0.15 | 57.32±0.05 / 55.52±0.04 | 62.53±0.42 / 61.88±0.57 | 63.21±0.34 / 62.79±0.42 | 62.80±0.06 / 62.24±0.10 |
| Day270 | 3 | 47.52±0.07 / 44.64±0.06 | 47.84±1.08 / 46.60±1.03 | 48.35±1.10 / 47.10±1.05 | 49.15±0.19 / 46.32±0.19 | 47.64±0.06 / 44.81±0.04 | 51.41±1.81 / 49.04±2.19 | 47.81±0.32 / 45.27±0.60 | 51.41±1.81 / 49.04±2.19 |
| Day270 | 10 | 47.52±0.07 / 44.64±0.06 | 59.12±0.53 / 58.20±0.63 | 60.96±0.33 / 60.44±0.30 | 49.84±0.04 / 47.10±0.07 | 47.66±0.08 / 44.83±0.06 | 59.26±0.23 / 58.20±0.34 | 60.96±0.33 / 60.44±0.30 | 60.41±0.84 / 59.74±1.08 |

## 10-shot 主诊断与选择频率

- Day14：DF 三个 seed 全选 `G_source`；VarCNNDirection 三个 seed 全选 `lambda=.01`。DF/VarCNNDirection 相对 G-source 为 0.000/+0.080 pp，分别比较强 current-only 高 1.951/4.881 pp。
- Day90：DF 三个 seed全选 `G_current`；VarCNNDirection 为 1 次 `G_current`、2 次 `alpha=.75`。恢复保留率为 DF 100.0%、VarCNNDirection 92.49%。
- Day270：DF 三个 seed全选 `G_current`；VarCNNDirection 为 2 次 `G_current`、1 次 current prototype。恢复保留率为 DF 100.0%、VarCNNDirection 95.55%。

选择频率随日期呈预期系统变化：Day14 全部落在不更新/强 source retention，Day90/270 的 12 次选择中 10 次为 current-only、2 次为 alpha=.75。它已经是比固定规则强得多的常规基线，但严格主门槛仍因 DF Day14 的 0.049 pp 边界差距失败。

跨家族 selected 相对 G-source 的 `corrected / harmed / net`（三个 seed mean±SD）如下。

| Backbone | 日期 | shot | corrected | harmed | net |
|---|---|---:|---:|---:|---:|
| DF | Day14 | 3 | 1.0±1.7 | 1.3±2.3 | -0.3±0.6 |
| DF | Day14 | 10 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| DF | Day90 | 3 | 1673.0±1101.3 | 1352.7±872.4 | +320.3±375.4 |
| DF | Day90 | 10 | 3595.3±23.2 | 1820.0±50.4 | +1775.3±71.0 |
| DF | Day270 | 3 | 1750.7±84.4 | 792.7±10.0 | +958.0±80.9 |
| DF | Day270 | 10 | 3508.0±74.5 | 1189.3±13.6 | +2318.7±66.5 |
| VarCNNDirection | Day14 | 3 | 243.7±169.1 | 281.0±193.8 | -37.3±38.4 |
| VarCNNDirection | Day14 | 10 | 206.3±9.1 | 194.7±12.1 | +11.7±4.9 |
| VarCNNDirection | Day90 | 3 | 928.7±1210.5 | 852.3±1038.8 | +76.3±183.1 |
| VarCNNDirection | Day90 | 10 | 3567.0±340.6 | 2056.7±344.8 | +1510.3±7.6 |
| VarCNNDirection | Day270 | 3 | 1781.0±514.3 | 1045.0±186.8 | +736.0±333.6 |
| VarCNNDirection | Day270 | 10 | 3909.0±130.6 | 1470.7±16.1 | +2438.3±146.4 |

## 3-shot selection noise

3-shot 跨家族选择与 posthoc oracle candidate 仅 **2/18（11.1%）** 一致，平均 macro-F1 regret **1.047±1.331 pp**，**7/18** 为候选不同且 regret>1 pp；预注册不稳定标记为 **17/18**。winner 的平均 fold SD 是 **4.732 pp**，fold winner 对最终 winner 平均一致率仅 **53.7%**。Day14 分散在 G-source 与四种 shrink 强度，Day90 分散在 alpha=.50/.75 与 shrink，Day270 才较一致地 5/6 选择 alpha=.50。

10-shot 明显更可靠：oracle candidate 一致率 **12/18（66.7%）**，平均 regret **0.221±0.543 pp**，只有 **1/18** 个实质选错；winner fold SD **3.126 pp**。但 winner-runner margin 相对 fold uncertainty 仍小，预注册不稳定判据把 18/18 都标为不稳定。这说明 10-shot 的性能 regret 小且日期选择结构清楚，但在 DF Day14 这种门槛边界上仍可能把 `G_source` 排在略优 shrinkage 之前。

## 与固定 alpha/lambda 的公平 common-query 比较

- 10-shot macro-F1 相对每单元较好的固定 rule：DF Day14 **-0.011 pp**、Day90 **+7.159 pp**、Day270 **+11.739 pp**；VarCNNDirection Day14 **+0.041 pp**、Day90 **+6.550 pp**、Day270 **+12.633 pp**，六单元平均 **+6.352 pp**。accuracy 平均改善 **+5.366 pp**。
- 3-shot macro-F1 六单元变化为 DF +0.011/+1.420/+3.394 pp，VarCNNDirection -0.188/+0.731/+2.721 pp，平均 **+1.348 pp**；accuracy 平均改善 **+1.121 pp**。

这不改写旧实验结论：旧固定规则仍是其原 query 定义下的正式结果；这里是为 common-query 公平比较而在本 run 重算。

## 失败归因与五个问题的直接回答

1. **普通 support-internal 选参是否解决主要权衡？** 严格否。它保留了两个 backbone 的后期恢复，也在 VarCNNDirection 保护 Day14，但 DF Day14 只因相对 current-only 的优势 1.951 pp 低于 2.0 pp 门槛而失败。
2. **10-shot 是否足够？** 对获得低-regret、随日期系统变化的强常规基线基本足够，但对预注册的跨 backbone 严格通过仍差一个边界单元，不能宣称已完全足够。
3. **3-shot selection noise 多大？** 很大：oracle 一致率 11.1%，regret 1.047±1.331 pp，7/18 实质选错、17/18 不稳定。
4. **相对固定 alpha/lambda 改善多少？** 10-shot 平均提高 6.352 pp macro-F1（5.366 pp accuracy），3-shot 提高 1.348 pp（1.121 pp accuracy）；主要改善来自 Day90/270 恢复。
5. **失败是 support 太少还是候选族不合适？还需新机制吗？** 10-shot query oracle envelope 在两个 backbone 上同时通过 Day14 与后期恢复，故不是 B；预注册归因为 **A selection failure**。3-shot 明确受 support 太少/高方差限制，10-shot 则是极小 oracle regret 仍足以触发边界门槛。现有证据没有证明必须进入新机制；它也没有达到“通过即暂停复杂机制”的严格条件。是否把这一强但严格未通过的常规基线作为停止点，仍由 Host 决定。

## 完整性与产物

- `metrics_long.csv` 288 行；`error_transfers.csv` 252 行；`selection_diagnostics.csv` 108 行；36 个 evaluation JSON 保存全部 candidate/selected 预测、CV fold 分数、拟合记录、common-query truth、选择与 posthoc oracle。
- 独立 verifier 重算 9,803,088 条 prediction rows 的 accuracy/macro-F1，检查 3/10-shot common query 相同、3-shot support 嵌套、完整 10-shot support 均从 query 排除、10-shot 三个 endpoint 与 donor 一致、冻结哈希、零训练/微调与零 checkpoint 输出；`integrity_check.json` 为 `passed=true`、0 errors、0 nonconverged shrink、0 G-current convergence warning。
- `posthoc_analysis.json` 保存逐 date/shot/backbone 的选择频率、accuracy/F1 fixed-rule 差值、oracle envelope 和 A/B 归因。中断恢复时保留了一份未登记、内容与正式 DF source prototype相同的 recovery 副本在 `logs/df_source_prototypes.interrupted.npy`；它未用于选择、拟合或评分。

限制：TemporalDrift 是已观察开发数据；oracle 只作归因。CV fold 只有 3 或 5 个且每类验证 1/2 条，fold 方差判据很敏感；严格结论依赖预注册的 2.0 pp Day14 边界，不应把 0.049 pp 的短缺解释成实质性大失败。
