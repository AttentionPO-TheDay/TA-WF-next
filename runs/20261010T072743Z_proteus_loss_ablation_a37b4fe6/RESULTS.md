# Proteus 损失组件消融结果

所有54个新任务完成并通过审计；历史36组预测注明复用，主归因相对本轮full_control。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| proteus_port | 70.793% | 72.251% | 8.512% | 22.143% |
| full_control | 70.909% | 72.255% | 8.489% | 21.992% |
| no_pseudo | 70.963% | 72.268% | 8.652% | 22.279% |
| no_mmd | 70.718% | 72.094% | 8.717% | 21.979% |
| no_entropy_diversity | 70.205% | 72.037% | 7.963% | 19.918% |
| no_source_ce | 71.012% | 72.371% | 9.534% | 24.075% |
| source_ce_only | 66.376% | 68.060% | 12.727% | 17.855% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.289% | 0.000% |
| proteus_port | 83.339% | 7.567% |
| full_control | 83.676% | 7.185% |
| no_pseudo | 83.337% | 7.590% |
| no_mmd | 83.578% | 7.228% |
| no_entropy_diversity | 84.681% | 5.833% |
| no_source_ce | 82.879% | 7.802% |
| source_ce_only | 82.169% | 8.788% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.459% | 0.000% |
| proteus_port | 70.207% | 8.183% |
| full_control | 69.611% | 8.817% |
| no_pseudo | 69.983% | 8.321% |
| no_mmd | 69.595% | 8.782% |
| no_entropy_diversity | 68.301% | 8.827% |
| no_source_ce | 70.015% | 9.824% |
| source_ce_only | 64.560% | 12.747% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.823% | 0.000% |
| proteus_port | 58.832% | 9.787% |
| full_control | 59.440% | 9.464% |
| no_pseudo | 59.569% | 10.046% |
| no_mmd | 58.981% | 10.140% |
| no_entropy_diversity | 57.632% | 9.229% |
| no_source_ce | 60.140% | 10.975% |
| source_ce_only | 52.398% | 16.644% |

```json
{
  "no_pseudo": {
    "macro_f1_delta_pp": 0.05391569577209987,
    "negative_delta_pp": 0.16327147406479392,
    "positive_cells": 4,
    "candidate_signal": false
  },
  "no_mmd": {
    "macro_f1_delta_pp": -0.19107386229451073,
    "negative_delta_pp": 0.22801979599187672,
    "positive_cells": 5,
    "candidate_signal": false
  },
  "no_entropy_diversity": {
    "macro_f1_delta_pp": -0.7041341023619974,
    "negative_delta_pp": -0.5255564959026946,
    "positive_cells": 3,
    "candidate_signal": false
  },
  "no_source_ce": {
    "macro_f1_delta_pp": 0.1026053784762948,
    "negative_delta_pp": 1.0447521948688574,
    "positive_cells": 5,
    "candidate_signal": false
  },
  "source_ce_only": {
    "macro_f1_delta_pp": -4.53324224889895,
    "negative_delta_pp": 4.2378776272776895,
    "positive_cells": 0,
    "candidate_signal": false
  }
}
```

no_pseudo仅删除伪标签监督损失；source_ce_only仍保留目标三路前向和BN更新。结果是当前wide+BN及开发日期的条件性损失贡献，不能直接归因于GMM本身或外推为论文RF复现。
新配对调度与上一轮端口不同。全部条件100 epoch/800步；历史10步adapter不是同预算控制。外部未来与其他因素未开放。
