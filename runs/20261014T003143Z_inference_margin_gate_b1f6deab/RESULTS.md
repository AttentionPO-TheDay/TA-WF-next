# 方案A：推断期安全门控结果

9个TemporalDrift query单元的离线阈值扫描完成。

| 设置 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| Exp-1 no_pseudo_fixed | 73.504% | 74.943% | 3.963% | 18.964% |
| Margin selected | 72.587% | 74.162% | 1.986% | 11.313% |
| Entropy selected | 72.553% | 74.083% | 1.858% | 10.261% |

全阈值曲线见artifacts/thresholds.csv；选择规则和标签权限见PLAN.md。
结果仅为TemporalDrift已观察开发证据。
