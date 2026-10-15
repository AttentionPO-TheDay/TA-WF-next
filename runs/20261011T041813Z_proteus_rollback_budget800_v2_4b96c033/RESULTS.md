# Proteus 800 候选更新无标签接受与回退结果

所有36项新适应任务完成；每项均尝试800次候选更新，回退只使用support无标签熵与高置信度比例。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 | 接受更新均值 |
|---|---:|---:|---:|---:|---:|
| frozen | 70.856% | 72.579% | 0.000% | 0.000% | 0.0 |
| full_control | 70.909% | 72.255% | 8.488% | 21.987% | 800.0 |
| rollback_10 | 72.456% | 73.845% | 6.112% | 21.031% | 152.2 |
| rollback_25 | 72.173% | 73.558% | 6.555% | 21.148% | 211.1 |
| rollback_50 | 71.589% | 72.963% | 7.094% | 20.605% | 255.6 |

## day14

| 条件 | Macro-F1 | 负适应率 | 接受更新 |
|---|---:|---:|---:|
| frozen | 86.290% | 0.000% | 0.0 |
| full_control | 83.676% | 7.183% | 800.0 |
| rollback_10 | 85.360% | 4.776% | 136.7 |
| rollback_25 | 84.688% | 5.594% | 208.3 |
| rollback_50 | 84.439% | 5.895% | 283.3 |

## day90

| 条件 | Macro-F1 | 负适应率 | 接受更新 |
|---|---:|---:|---:|
| frozen | 69.454% | 0.000% | 0.0 |
| full_control | 69.611% | 8.816% | 800.0 |
| rollback_10 | 71.301% | 6.291% | 163.3 |
| rollback_25 | 70.817% | 6.665% | 208.3 |
| rollback_50 | 70.439% | 7.150% | 200.0 |

## day270

| 条件 | Macro-F1 | 负适应率 | 接受更新 |
|---|---:|---:|---:|
| frozen | 56.824% | 0.000% | 0.0 |
| full_control | 59.440% | 9.464% | 800.0 |
| rollback_10 | 60.706% | 7.268% | 156.7 |
| rollback_25 | 61.014% | 7.406% | 216.7 |
| rollback_50 | 59.888% | 8.238% | 283.3 |

```json
{
  "rollback_10": {
    "macro_f1_delta_pp": 1.5466583458903393,
    "negative_delta_pp": -2.375947120806087,
    "positive_cells": 9,
    "candidate_signal": true
  },
  "rollback_25": {
    "macro_f1_delta_pp": 1.2638639473276703,
    "negative_delta_pp": -1.9326479649488304,
    "positive_cells": 9,
    "candidate_signal": true
  },
  "rollback_50": {
    "macro_f1_delta_pp": 0.6797474102762768,
    "negative_delta_pp": -1.393190946709634,
    "positive_cells": 7,
    "candidate_signal": true
  }
}
```

当前结论仅限 TemporalDrift 已观察开发数据和 wide+BN 上的 Proteus 移植；不等同官方 RF 复现。
