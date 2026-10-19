# 实验结果

纯净源域三 seed × Day14/90/270 × 五条件共 45 单元已完成；其中 9 个 frozen 单元引用独立审计通过的历史 run，本 run 新增 36 个适应单元。`postrun_audit.json` 已逐项重算并重载全部新 checkpoint。TemporalDrift 是已观察开发集。

| 条件 | Macro-F1 (%) | Accuracy (%) | NAR (%) | Recovery (%) |
|---|---:|---:|---:|---:|
| frozen | 72.645 | 74.390 | 0.000 | 0.000 |
| tent_bn | 72.543 | 74.370 | 3.730 | 10.807 |
| tent_entropy_gate | 72.969 | 74.803 | 2.101 | 7.629 |
| proteus_default | 72.324 | 73.579 | 8.497 | 21.787 |
| no_pseudo_fixed | 74.755 | 76.275 | 3.708 | 17.786 |

逐 seed、日期、条件结果在 `artifacts/scores.json`；800 步支持集动力学、预测和成本留在本 run 的 artifacts/checkpoints。TENT BN 的 query 前向使用批统计量，信息权限与另外四条件不同，单独解释。外部数据未用于此矩阵。
