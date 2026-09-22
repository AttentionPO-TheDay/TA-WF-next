# 实验结果

## 完成状态

Frozen Global Representation Temporal Recoverability Diagnostic 已按 `RECOVERABILITY_PLAN.md` 完成。新增 backbone 训练/微调次数为 0；只读复用 `exp_9121b664a1854097` 的 DF epoch 29 与 VarCNNDirection epoch 23 正式 checkpoint。未改写计划、support manifest、隔离审计、donor checkpoint、正式 split 或旧实验 artifact；未使用外部数据、无标签 TTA、DNNF/TFAN、表示更新或多日期记忆。

内容审计先于拟合和评分完成。Day14/90/270 分别保留 22,603/28,599/19,935 条唯一 canonical trace，每类最少 201/239/85 条；三个日期均无 current 内重复、source-role 内容重叠或跨日期内容交集。每个 seed 的 3-shot（306 条）严格嵌套于 10-shot（1,020 条），support/query 的 row 与 admitted-input 内容交集均为 0。

## Provenance 与 source 线性头

| Backbone | Checkpoint / best epoch | SHA-256 | 最终 embedding | G-source selected C | source-validation F1 |
|---|---|---|---|---:|---:|
| DF | `runs/exp_9121b664a1854097/checkpoints/df_best.pt`, epoch 29 | `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae` | `DF.forward_features` / `mlp` 输入，512 维 | 10 | 73.16% |
| VarCNNDirection | `runs/exp_9121b664a1854097/checkpoints/varcnn_direction_best.pt`, epoch 23 | `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83` | `VarCNNDirection.forward_features` / `mlp` 输入，512 维 | 10 | 72.12% |

G-source 仅拟合 source supervised_train，C 仅由 official source validation macro-F1 选择。G-current 从头仅拟合对应 current support，直接继承同 backbone 的 G-source C；simple-maintenance 是纯 current normalized class prototype，无 source prototype 或插值系数。query 标签只在该日期全部预测固定后用于评分。

## 绝对性能与恢复量

单元格为三个 support seed 的 `accuracy ± SD / macro-F1 ± SD`，均为百分数。恢复量列为 macro-F1 百分点。

### DF

| 日期 | shot | A | G-source | G-current | Simple | G-current−A | G-current−G-source | Simple−A | Simple−G-current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day14 | 3 | 71.25±0.05 / 70.78±0.05 | 72.50±0.05 / 72.34±0.05 | 64.65±1.48 / 64.04±1.50 | 64.50±1.53 / 63.87±1.50 | -6.74±1.55 | -8.29±1.55 | -6.91±1.55 | -0.17±0.31 |
| Day14 | 10 | 71.32±0.08 / 70.85±0.09 | 72.56±0.07 / 72.39±0.07 | 70.67±0.13 / 70.44±0.16 | 70.12±0.12 / 69.81±0.15 | -0.40±0.15 | -1.95±0.17 | -1.04±0.23 | -0.63±0.24 |
| Day90 | 3 | 56.55±0.01 / 54.48±0.01 | 57.27±0.01 / 55.44±0.00 | 55.76±0.59 / 54.65±0.75 | 55.26±0.64 / 54.14±0.87 | +0.17±0.76 | -0.79±0.75 | -0.35±0.88 | -0.51±0.23 |
| Day90 | 10 | 56.56±0.03 / 54.50±0.03 | 57.28±0.03 / 55.45±0.02 | 63.72±0.24 / 63.25±0.21 | 62.25±0.16 / 61.54±0.14 | +8.75±0.22 | +7.80±0.23 | +7.04±0.12 | -1.71±0.25 |
| Day270 | 3 | 46.10±0.01 / 43.16±0.01 | 46.70±0.01 / 43.90±0.01 | 49.50±1.08 / 47.89±0.97 | 49.02±0.93 / 47.28±0.88 | +4.73±0.96 | +3.99±0.96 | +4.12±0.88 | -0.61±0.09 |
| Day270 | 10 | 46.12±0.05 / 43.18±0.05 | 46.71±0.01 / 43.90±0.01 | 58.97±0.34 / 58.15±0.45 | 57.25±0.01 / 55.99±0.26 | +14.98±0.49 | +14.26±0.45 | +12.81±0.28 | -2.17±0.44 |

### VarCNNDirection

| 日期 | shot | A | G-source | G-current | Simple | G-current−A | G-current−G-source | Simple−A | Simple−G-current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day14 | 3 | 67.75±0.05 / 67.48±0.05 | 70.85±0.06 / 70.70±0.06 | 55.04±2.69 / 54.31±2.76 | 55.19±2.76 / 54.43±2.89 | -13.17±2.81 | -16.39±2.82 | -13.05±2.95 | +0.12±0.25 |
| Day14 | 10 | 67.81±0.06 / 67.54±0.05 | 70.89±0.05 / 70.75±0.05 | 66.25±0.53 / 65.95±0.58 | 65.74±0.74 / 65.28±0.75 | -1.59±0.58 | -4.80±0.59 | -2.26±0.75 | -0.67±0.17 |
| Day90 | 3 | 54.38±0.01 / 53.02±0.01 | 57.31±0.00 / 55.46±0.00 | 52.08±0.35 / 51.16±0.39 | 51.29±0.22 / 50.39±0.27 | -1.86±0.39 | -4.30±0.40 | -2.63±0.27 | -0.77±0.30 |
| Day90 | 10 | 54.42±0.04 / 53.07±0.04 | 57.32±0.07 / 55.47±0.05 | 63.21±0.34 / 62.79±0.42 | 61.43±0.51 / 60.87±0.62 | +9.73±0.45 | +7.32±0.47 | +7.81±0.65 | -1.92±0.26 |
| Day270 | 3 | 44.45±0.03 / 41.77±0.03 | 47.58±0.05 / 44.71±0.05 | 48.39±1.09 / 47.14±1.07 | 47.86±1.12 / 46.61±1.08 | +5.37±1.10 | +2.43±1.11 | +4.84±1.10 | -0.53±0.02 |
| Day270 | 10 | 44.40±0.07 / 41.73±0.06 | 47.52±0.07 / 44.64±0.06 | 60.96±0.33 / 60.44±0.30 | 59.12±0.53 / 58.20±0.63 | +18.71±0.27 | +15.80±0.31 | +16.47±0.60 | -2.24±0.34 |

## 10-shot 相对 3-shot

G-current macro-F1 的配对增益在 DF Day14/90/270 为 +6.40/+8.60/+10.27 pp，在 VarCNNDirection 为 +11.64/+11.63/+13.30 pp；三个 seed 的完整 SD 和全部方法见 `artifacts/summary.json`。两 backbone 的三个日期均超过预注册的 +2 pp 实质增益门槛。

## 对三个问题的直接回答

1. **冻结最终 global embedding 是否可恢复？** 有条件地是。按预注册门槛，10-shot 在两个 backbone 的 Day90 和 Day270、全部三个 seed 上稳定超过 A，分别恢复 +8.75/+14.98 pp（DF）与 +9.73/+18.71 pp（VarCNNDirection）macro-F1；同时相对 G-source 仍有 +7.80/+14.26 与 +7.32/+15.80 pp，说明收益来自当前监督信息，而不只是把原头换成线性分类器。Day14 的 A/G-source 本来较强，current-only 读出即使 10-shot 仍略低于 A，故恢复并非所有日期普遍成立。这只说明冻结表示在少量当前监督下仍有可利用信息，不说明表示天然抗漂移。
2. **需要多少当前监督？** 本协议下需要约 10 条/类才得到跨 backbone、跨多个日期的稳定恢复。3-shot 未通过双 backbone 门槛：DF 没有日期达到“全部 seed 为正且 mean≥5 pp”，VarCNNDirection 只有 Day270 达到；从 3 增至 10 条/类在所有日期和两个 backbone 都带来明显增益。
3. **简单维护是否接近线性读出的恢复？** 是，按预注册聚合门槛两个 backbone 均通过。Simple 相对 G-current 跨全部日期/shot/seed 的平均 macro-F1 差为 DF -0.97 pp、VarCNNDirection -1.00 pp，三个日期均满足预注册的日期聚合接近条件。需要保留的细节是 10-shot Day270 单独差距为 -2.17/-2.24 pp，略超过 2 pp；因此 simple prototype 是强基线，但不能称为每个单元都等价。

## 完整性、产物与限制

- `artifacts/metrics_long.csv`：144 行（36 配置×4 方法）；`artifacts/summary.json`：seed 均值/样本 SD、全部恢复量、配对 shot 增益和预注册判断。
- 36 个 `backbone_date_seed_shot.json` 保存 query row/truth、四种预测、accuracy/macro precision/recall/F1、102 类逐网站指标、manifest 哈希和拟合记录。
- `artifacts/integrity_check.json` 独立复核 36 个评价文件、3,319,104 条 method prediction、相同 query/truth、预测长度、重算 accuracy/F1、102 类覆盖、source/model/manifest/checkpoint 哈希、零 backbone 训练和 query 权限记录；`passed=true`，无错误、无 G-current convergence warning、无临时文件。
- CPU 冻结特征抽取仅影响耗时，不改变模型或研究定义。G-current 是 current-only 小样本监督诊断，不是性能上界或可直接包装的方法贡献；本实验没有与匹配标签预算的表示更新比较。

实际证据支持将 simple current prototype 作为后续任何输出端维护研究的强基线。是否开启后续实验由 Host 决定；本实验不自动扩展。
