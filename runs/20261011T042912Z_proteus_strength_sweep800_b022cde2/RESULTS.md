# Proteus 800 候选更新适应强度结果

所有36项新任务完成；四个强度只改变固定 Adam 学习率倍率，query 标签仅用于最终评分。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.856% | 72.579% | 0.000% | 0.000% |
| strength_025 | 73.432% | 74.908% | 4.052% | 19.058% |
| strength_050 | 73.059% | 74.493% | 5.308% | 20.929% |
| strength_075 | 71.626% | 72.958% | 7.753% | 22.218% |
| strength_100 | 70.909% | 72.255% | 8.488% | 21.987% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.290% | 0.000% |
| strength_025 | 86.645% | 2.796% |
| strength_050 | 86.366% | 3.524% |
| strength_075 | 84.723% | 5.815% |
| strength_100 | 83.676% | 7.183% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.454% | 0.000% |
| strength_025 | 72.151% | 4.231% |
| strength_050 | 71.523% | 5.836% |
| strength_075 | 70.004% | 8.561% |
| strength_100 | 69.611% | 8.816% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.824% | 0.000% |
| strength_025 | 61.499% | 5.128% |
| strength_050 | 61.289% | 6.564% |
| strength_075 | 60.151% | 8.882% |
| strength_100 | 59.440% | 9.464% |

```json
{
  "strength_025": {
    "macro_f1_delta_pp": 2.5229272722299356,
    "negative_delta_pp": -4.43592806547461,
    "positive_cells": 9,
    "candidate_signal": true
  },
  "strength_050": {
    "macro_f1_delta_pp": 2.1503821344047935,
    "negative_delta_pp": -3.1795262772485637,
    "positive_cells": 9,
    "candidate_signal": true
  },
  "strength_075": {
    "macro_f1_delta_pp": 0.7171404525688652,
    "negative_delta_pp": -0.7349917107081148,
    "positive_cells": 7,
    "candidate_signal": false
  }
}
```

结论限于 TemporalDrift 已观察开发数据和当前 wide+BN 上的 Proteus 移植。
