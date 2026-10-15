# 图片实验结果

9个新适应任务与27条历史参考完成，postrun审计通过。历史参考不计独立重复。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| no_pseudo_fixed | 73.504% | 74.943% | 3.963% | 18.964% |
| dynamic_lr_only | 73.477% | 74.932% | 3.986% | 18.946% |
| soft_pacing | 72.566% | 74.347% | 4.212% | 17.855% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.289% | 0.000% |
| no_pseudo_fixed | 86.642% | 2.757% |
| dynamic_lr_only | 86.643% | 2.737% |
| soft_pacing | 86.219% | 3.256% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.459% | 0.000% |
| no_pseudo_fixed | 72.346% | 4.109% |
| dynamic_lr_only | 72.344% | 4.102% |
| soft_pacing | 71.602% | 4.188% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.823% | 0.000% |
| no_pseudo_fixed | 61.524% | 5.025% |
| dynamic_lr_only | 61.443% | 5.119% |
| soft_pacing | 59.876% | 5.192% |

适应秒/任务 207.958；反向秒/任务 160.320；query摊销毫秒/trace 0.145；峰值分配GiB 2.314；更新数 800。

负适应率=静态模型原本正确、适应后错误的样本数 / 静态正确样本数。纠错率=适应后纠正的样本数 / 静态错误样本数。
support-only batch1/128计时、逐更新控制/损失记录见artifacts；共享GPU上仅报告描述性时间，不作跨批次速度倍率结论。
仅TemporalDrift已观察开发证据。目标数据不是按时间排序流，Exp-4仅检验适应优化过程的窗口反馈，不能声称检测了真实在线漂移速率。
EMA与一致性为已有通用思想；当前实验只检验本模型收益，不证明论文创新。
初版错误与保留产物见repair_history.md。
{
  "soft_pacing": {
    "macro_f1": 0.7256567055892863,
    "negative_adaptation_rate": 0.042120807850562686,
    "f1_delta_vs_fixed_pp": -0.9381560319289273,
    "negative_delta_vs_fixed_pp": 0.24861714584663297,
    "positive_paired_cells": 1,
    "screen_pass": false,
    "aspirational_goal_pass": false
  }
}
