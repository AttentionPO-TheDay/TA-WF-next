# 实验结果

## 完成状态

首轮冻结 screening 已完成。只新增两次训练：DF seed 6238 与方向单分支 `VarCNNDirection` seed 6238；各跑满 30 epoch，B/C/D/E 共用各自 backbone 的同一个 source-only best checkpoint。完整评价覆盖 source-holdout、Day14、Day30、Day90、Day150、Day270。未扩展 seed、模型、数据集、few-shot、网格或机制。

执行前完整读取并核对了指定 PLAN/RUN_COMMANDS、项目协议和冻结 artifact，详情见 `execution_requirements.md`。正式 split 直接只读使用旧实验的 `splits_v3.json`，SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`；全部 7 个 NPZ 的 SHA-256 与 split manifest 一致。旧 experiment 未被写入。

冻结方案与用户概括的差异：当前 PLAN 是评价 revision 4，除 A/B/C/D 外还要求 E 同层全局对照以及 D-E；本次按 PLAN 完成 A/B/C/D/E。原入口硬编码旧 run 路径，执行前仅增加显式 `--run-dir` 隔离输出，split 与 source-only v4 audit 仍从旧目录只读，未改变研究定义。

## 训练与选模

| Backbone | 训练 epoch | Best epoch | source-valid accuracy | source-valid macro-F1 |
|---|---:|---:|---:|---:|
| DF | 30 | 29 | 72.407% | 71.655% |
| VarCNNDirection | 30 | 23 | 69.630% | 69.142% |

两者均从头初始化，seed 6238，AdamW(lr=0.001, weight_decay=0.0001)，batch 64，仅按 official source validation macro-F1 选模；没有未来标签参与训练或选模。

## A/B/C/D/E 主要结果

单元格为 accuracy / macro-F1，均为百分数。完整 macro precision/recall、逐网站 accuracy 见 `artifacts/metrics_long.csv` 与两个 `*_evaluation.json`。

| Backbone | 日期 | A | B | C | D | E |
|---|---|---:|---:|---:|---:|---:|
| DF | source | 71.96 / 71.35 | 58.14 / 56.65 | 52.16 / 50.88 | 5.54 / 4.96 | 3.97 / 3.92 |
| DF | Day14 | 71.26 / 70.79 | 59.24 / 58.25 | 53.31 / 52.34 | 6.05 / 5.77 | 4.69 / 4.59 |
| DF | Day30 | 65.10 / 64.05 | 54.53 / 53.04 | 49.29 / 48.09 | 5.90 / 5.65 | 4.62 / 4.52 |
| DF | Day90 | 56.55 / 54.48 | 46.56 / 44.36 | 41.58 / 39.47 | 5.16 / 5.03 | 4.43 / 4.45 |
| DF | Day150 | 50.50 / 47.59 | 42.02 / 39.31 | 37.84 / 35.30 | 4.77 / 4.43 | 4.05 / 3.95 |
| DF | Day270 | 46.07 / 43.15 | 38.39 / 35.48 | 34.70 / 32.01 | 4.64 / 4.33 | 3.96 / 3.55 |
| VarCNNDirection | source | 68.73 / 68.48 | 46.67 / 45.38 | 43.04 / 42.00 | 7.16 / 6.11 | 5.20 / 5.12 |
| VarCNNDirection | Day14 | 67.77 / 67.50 | 47.79 / 46.75 | 41.57 / 40.45 | 6.13 / 5.58 | 4.87 / 4.63 |
| VarCNNDirection | Day30 | 61.98 / 61.41 | 44.22 / 43.13 | 38.64 / 37.65 | 5.95 / 5.36 | 5.20 / 4.93 |
| VarCNNDirection | Day90 | 54.37 / 53.01 | 37.81 / 36.26 | 32.95 / 31.69 | 5.37 / 4.99 | 4.83 / 4.72 |
| VarCNNDirection | Day150 | 48.77 / 46.20 | 33.76 / 31.62 | 29.45 / 27.54 | 4.97 / 4.39 | 4.26 / 4.05 |
| VarCNNDirection | Day270 | 44.41 / 41.73 | 31.78 / 29.37 | 28.54 / 26.42 | 4.76 / 4.32 | 4.30 / 3.80 |

五个 future 日期的未加权均值：DF 的 A/B/C/D/E accuracy 分别为 57.90/48.15/43.34/5.30/4.35%，VarCNNDirection 为 55.46/39.07/34.23/5.44/4.69%。

## D-C、D-E 与相对 source 差

单元格为 accuracy pp；macro-F1 版本及全精度值见 `artifacts/deltas.csv`。

| Backbone | 日期 | D-C | future(D-C)-source(D-C) | D-E | future(D-E)-source(D-E) |
|---|---|---:|---:|---:|---:|
| DF | source | -46.62 | — | +1.57 | — |
| DF | Day14 | -47.26 | -0.64 | +1.36 | -0.21 |
| DF | Day30 | -43.39 | +3.23 | +1.28 | -0.29 |
| DF | Day90 | -36.42 | +10.20 | +0.73 | -0.84 |
| DF | Day150 | -33.07 | +13.54 | +0.72 | -0.85 |
| DF | Day270 | -30.06 | +16.56 | +0.68 | -0.89 |
| VarCNNDirection | source | -35.88 | — | +1.96 | — |
| VarCNNDirection | Day14 | -35.44 | +0.44 | +1.26 | -0.70 |
| VarCNNDirection | Day30 | -32.69 | +3.19 | +0.75 | -1.21 |
| VarCNNDirection | Day90 | -27.58 | +8.30 | +0.55 | -1.41 |
| VarCNNDirection | Day150 | -24.48 | +11.40 | +0.71 | -1.25 |
| VarCNNDirection | Day270 | -23.79 | +12.10 | +0.46 | -1.50 |

相对 source 的 D-C 差在后期变正，仅因为 C 随漂移下降得比接近地板的 D 更快，不能解释为 D 获得绝对收益。D 在所有 backbone × date 上均显著低于 C。D 对 E 始终为小幅正值，但绝对 D accuracy 只有 4.64–7.16%，且 future 相对 source 的 D-E accuracy 增量全部为负。

## 局部匹配、共同错误与区分性

所有样本都有至少 4 个有效浅层位置；两 backbone、六个域的 D/E fallback 均为 0。DF 的 future D 正确同站匹配 6,265，错误跨站匹配 111,803；VarCNNDirection 分别为 6,427 与 111,641。单个 reference 在一个域内可被 90–99 个不同真网站命中，单个 reference-region 可被 90–97 个真网站命中。

跨 backbone 对齐同一 query 后：五个 future 日期共 118,068 条样本，其中 56,821 条的 C 在两个 backbone 都错误；40,612 条得到两个 backbone 相同的错误 D 类别；20,332 条同时满足“两 backbone 的 C 都错且 D 给出同一错误类别”。例如聚合六个域后，真类 36→D 类 96 在 DF/VarCNNDirection 中分别出现 124/251 次，82→9 为 199/141 次，45→29 为 183/155 次，36→18 为 173/164 次。

两个 backbone 各自 future top-50 高频 reference-region 模式中有 34 个共同位置。最频繁共同模式包括 `(reference_position, region)` 55/2、36/1、188/3：DF 分别命中 5,300/4,563/3,687 次，VarCNNDirection 为 5,815/6,337/6,277 次；这些模式覆盖 96–102 个不同真网站，主导真类比例仅约 3.5–6.5%。这是跨 backbone 可重复、但类别区分性很低的共同局部模式。完整 hit breadth、频率、entropy、top true sites 和每域共同错误见 `artifacts/posthoc_diagnostics.json`。

## Reference 存储与推理成本

两模型均使用 204 条 reference。按评价入口实际 float32 表示：原始输入 4,080,000 bytes；全局 reference 417,792 bytes；B 若只保留 102 个类原型等价 208,896 bytes。DF 的 D 四区域 104,448 bytes、E 同层均值 26,112 bytes；VarCNNDirection 分别为 208,896 与 52,224 bytes。两者 eligible reference 均为 204、102 类完整。

单次 reference 抽取 wall time：DF 0.573 s，VarCNNDirection 0.601 s。全部 120,108 个 query（source 加五个 future）的共享单 encoder 抽取 + 五分类器计时合计：DF 23.03 s，VarCNNDirection 26.48 s。每 query 等价 score 数：A 102 logits、B 102 prototype cosines、C 204 reference cosines、D 3,264 region cosines（204×16）、E 204 reference cosines。逐域拆分见评价 JSON；该计时为本次单次 GPU wall-time 统计，不是重复 benchmark。

## 第一轮证据判断（不替 Host 作研究决策）

- “C 是否已经解释 D 的收益”：不适用；D 没有相对 C 的收益，所有 D-C 均大幅为负。
- “D 是否只在一个 backbone 有效”：否；按相对 C 的预定问题，两个 backbone 都无效。
- “D 是否在两个 backbone 都有 C 之外的稳定价值”：没有证据。D-E 虽一致小幅为正，但规模很小、绝对性能接近地板，且 future 相对 source 的 D-E accuracy 增量全为负。
- “是否存在共同且可重复的错误匹配”：是；同 query 的共同错误、重复 true→D 类对和 top reference-region 重合都很明显，并表现为低类别区分性。
- 本轮实际结果支持的最窄结论是：冻结的简单浅层四区域 D 不足以支持继续开发该局部机制本身。是否开启新的研究方向属于 Host 的后续决定，本实验不自动扩展。

## 产物、验证与限制

- 主表：`artifacts/metrics_long.csv`（60 行结果）；差值：`artifacts/deltas.csv`；跨 backbone 诊断：`artifacts/posthoc_diagnostics.json`。
- 每 backbone 的评价 JSON 含主要指标、102 类逐网站 accuracy、覆盖/回退、局部匹配诊断和成本；每 backbone × domain 的 predictions JSON 含 row ID、冻结预测、D reference/region 选择、有效位置数以及 lossless packbits 实际位置 mask。
- checkpoint SHA-256：DF `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`；VarCNNDirection `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83`。
- 代码验证：`py_compile` 通过；单元测试 9 passed、1 optional legacy parity skipped；`scripts/experiment.py check` 通过；`git diff --check` 通过。
- 固有限制：NPZ 无 session/site 字符串 ID；official valid 内有 9 个固定重复副本；VarCNNDirection 是明确标注的方向单分支派生；区域 RF 高度重叠，不能解释为独立网页资源；未来标签用于已冻结预测后的开发评分与事后诊断，因此这些日期不是独立确认数据。
