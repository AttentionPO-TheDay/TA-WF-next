# Proteus 低学习率损失组件归因结果

所有54个新任务完成并通过审计；主归因相对本轮低学习率 full_control，历史36组预测明确标记复用。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| proteus_port | 70.793% | 72.251% | 8.512% | 22.143% |
| full_control | 73.432% | 74.908% | 4.049% | 19.055% |
| no_pseudo | 73.493% | 74.922% | 4.081% | 19.136% |
| no_mmd | 73.348% | 74.849% | 4.037% | 18.806% |
| no_entropy_diversity | 72.378% | 74.174% | 3.761% | 16.083% |
| no_source_ce | 72.497% | 73.932% | 6.296% | 21.166% |
| source_ce_only | 71.033% | 72.764% | 4.578% | 13.448% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.289% | 0.000% |
| proteus_port | 83.339% | 7.567% |
| full_control | 86.645% | 2.796% |
| no_pseudo | 86.645% | 2.778% |
| no_mmd | 86.629% | 2.780% |
| no_entropy_diversity | 86.826% | 2.494% |
| no_source_ce | 85.092% | 4.743% |
| source_ce_only | 86.621% | 2.562% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.459% | 0.000% |
| proteus_port | 70.207% | 8.183% |
| full_control | 72.151% | 4.235% |
| no_pseudo | 72.342% | 4.158% |
| no_mmd | 72.061% | 4.211% |
| no_entropy_diversity | 70.487% | 3.970% |
| no_source_ce | 71.473% | 6.464% |
| source_ce_only | 69.252% | 4.726% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.823% | 0.000% |
| proteus_port | 58.832% | 9.787% |
| full_control | 61.499% | 5.116% |
| no_pseudo | 61.493% | 5.307% |
| no_mmd | 61.355% | 5.119% |
| no_entropy_diversity | 59.821% | 4.819% |
| no_source_ce | 60.927% | 7.679% |
| source_ce_only | 57.227% | 6.445% |

```json
{
  "no_pseudo": {
    "macro_f1_delta_pp": 0.0611384040560492,
    "negative_delta_pp": 0.03224619368190906,
    "positive_cells": 6,
    "candidate_signal": false
  },
  "no_mmd": {
    "macro_f1_delta_pp": -0.08353990884063522,
    "negative_delta_pp": -0.01226073016087001,
    "positive_cells": 1,
    "candidate_signal": false
  },
  "no_entropy_diversity": {
    "macro_f1_delta_pp": -1.0537250111735785,
    "negative_delta_pp": -0.2881031118058902,
    "positive_cells": 3,
    "candidate_signal": false
  },
  "no_source_ce": {
    "macro_f1_delta_pp": -0.9343461943526865,
    "negative_delta_pp": 2.2466023437832376,
    "positive_cells": 0,
    "candidate_signal": false
  },
  "source_ce_only": {
    "macro_f1_delta_pp": -2.398390669598627,
    "negative_delta_pp": 0.5287249791011784,
    "positive_cells": 2,
    "candidate_signal": false
  }
}
```

no_pseudo仅删除伪标签监督损失；source_ce_only仍保留目标三路前向和BN更新。结果是当前wide+BN及开发日期的条件性损失贡献，不能直接归因于GMM本身或外推为论文RF复现。
新配对调度与上一轮端口不同。全部条件100 epoch/800步；历史10步adapter不是同预算控制。外部未来与其他因素未开放。

## 审计补充

原始执行因历史结果复制计数错误退出；54 项训练本身均完成。筛选修复后，`artifacts/audit.json` 显示 90 条结果、54 个新 checkpoint、1080 个 GMM 文件、54 次 checkpoint 重载和冻结 hash 均通过。修复过程见 `repair_history.md`，原始 144 条文件保留在 `artifacts/scores_pre_repair_144_rows.json`。
