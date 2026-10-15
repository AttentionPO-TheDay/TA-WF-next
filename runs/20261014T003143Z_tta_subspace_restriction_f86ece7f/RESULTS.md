# 图片实验结果

18个新适应任务与27条历史参考完成，postrun审计通过。历史参考不计独立重复。初始进度事件误写为9项，已按实际两作用域×3 seed×3日期修正为18项；原始事件保留。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| no_pseudo_fixed | 73.504% | 74.943% | 3.963% | 18.964% |
| dynamic_lr_only | 73.477% | 74.932% | 3.986% | 18.946% |
| wide_bn_only | 72.146% | 73.779% | 4.406% | 16.081% |
| attention_bn_only | 73.053% | 74.407% | 4.411% | 18.122% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.289% | 0.000% |
| no_pseudo_fixed | 86.642% | 2.757% |
| dynamic_lr_only | 86.643% | 2.737% |
| wide_bn_only | 86.413% | 2.811% |
| attention_bn_only | 86.392% | 2.884% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.459% | 0.000% |
| no_pseudo_fixed | 72.346% | 4.109% |
| dynamic_lr_only | 72.344% | 4.102% |
| wide_bn_only | 70.514% | 4.571% |
| attention_bn_only | 72.203% | 4.457% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.823% | 0.000% |
| no_pseudo_fixed | 61.524% | 5.025% |
| dynamic_lr_only | 61.443% | 5.119% |
| wide_bn_only | 59.510% | 5.836% |
| attention_bn_only | 60.564% | 5.892% |

适应秒/任务 144.063；反向秒/任务 56.516；query摊销毫秒/trace 0.153；峰值分配GiB 2.006；更新数 800。

适应秒/任务 122.290；反向秒/任务 31.889；query摊销毫秒/trace 0.173；峰值分配GiB 2.110；更新数 800。

负适应率=静态模型原本正确、适应后错误的样本数 / 静态正确样本数。纠错率=适应后纠正的样本数 / 静态错误样本数。
support-only batch1/128计时、逐更新控制/损失记录见artifacts；共享GPU上仅报告描述性时间，不作跨批次速度倍率结论。
仅TemporalDrift已观察开发证据。目标数据不是按时间排序流，Exp-4仅检验适应优化过程的窗口反馈，不能声称检测了真实在线漂移速率。
EMA与一致性为已有通用思想；当前实验只检验本模型收益，不证明论文创新。
初版错误与保留产物见repair_history.md。
{
  "wide_bn_only": {
    "macro_f1": 0.7214586396485534,
    "negative_adaptation_rate": 0.04406084160180482,
    "f1_delta_vs_fixed_pp": -1.3579626260022182,
    "negative_delta_vs_fixed_pp": 0.44262052097084614,
    "positive_paired_cells": 1,
    "screen_pass": false,
    "aspirational_goal_pass": false
  },
  "attention_bn_only": {
    "macro_f1": 0.7305290944294004,
    "negative_adaptation_rate": 0.04410862204617019,
    "f1_delta_vs_fixed_pp": -0.4509171479175178,
    "negative_delta_vs_fixed_pp": 0.44739856540738365,
    "positive_paired_cells": 0,
    "screen_pass": false,
    "aspirational_goal_pass": false
  }
}
