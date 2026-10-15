# TemporalDrift MVI 结果

45个新任务和27条历史参考完成；独立审计通过。主候选为mvi_joint；不根据最好结果重定义主候选。

| 条件 | Macro-F1 | Accuracy | 负适应率 |
|---|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% |
| proteus_strength025 | 73.432% | 74.908% | 4.049% |
| rollback10 | 72.456% | 73.845% | 6.111% |
| no_pseudo_fixed | 73.504% | 74.943% | 3.963% |
| dynamic_lr_only | 73.477% | 74.932% | 3.986% |
| bn_interp_only | 73.504% | 74.943% | 3.963% |
| mvi_joint | 73.478% | 74.933% | 3.986% |
| mvi_joint_skip | 69.659% | 71.166% | 9.277% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.289% | 0.000% |
| proteus_strength025 | 86.645% | 2.796% |
| rollback10 | 85.360% | 4.775% |
| no_pseudo_fixed | 86.642% | 2.757% |
| dynamic_lr_only | 86.643% | 2.737% |
| bn_interp_only | 86.642% | 2.757% |
| mvi_joint | 86.645% | 2.735% |
| mvi_joint_skip | 83.662% | 6.440% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.459% | 0.000% |
| proteus_strength025 | 72.151% | 4.235% |
| rollback10 | 71.301% | 6.292% |
| no_pseudo_fixed | 72.346% | 4.109% |
| dynamic_lr_only | 72.344% | 4.102% |
| bn_interp_only | 72.346% | 4.109% |
| mvi_joint | 72.346% | 4.105% |
| mvi_joint_skip | 68.449% | 9.453% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.823% | 0.000% |
| proteus_strength025 | 61.499% | 5.116% |
| rollback10 | 60.706% | 7.265% |
| no_pseudo_fixed | 61.524% | 5.025% |
| dynamic_lr_only | 61.443% | 5.119% |
| bn_interp_only | 61.524% | 5.025% |
| mvi_joint | 61.444% | 5.116% |
| mvi_joint_skip | 56.866% | 11.938% |

## 计算成本

| 条件 | 适应秒/任务 | 反向秒/任务 | target暴露数 | 跳过更新 | 峰值显存GiB |
|---|---:|---:|---:|---:|---:|
| no_pseudo_fixed | 64.559 | 30.990 | 102400.0 | 0.0 | 2.294 |
| dynamic_lr_only | 64.362 | 30.996 | 102400.0 | 0.0 | 2.294 |
| bn_interp_only | 64.546 | 31.022 | 102400.0 | 0.0 | 2.294 |
| mvi_joint | 64.402 | 31.034 | 102400.0 | 0.0 | 2.294 |
| mvi_joint_skip | 54.414 | 21.999 | 13935.3 | 0.0 | 1.645 |

逐条控制记录、entropy_gate、support-only batch1/128推理p50/p95见artifacts。GPU共享；反向耗时是batch摊销，不是单条在线时延。静态/历史对照成本不能跨硬件直接作速度倍数比较。

当前输入依赖时间戳；WTT/AWF及非时间因素尚未评分。完全去除伪标签同时移除第三路前向，不等同上一轮no_pseudo损失消融。

{
  "no_pseudo_fixed": {
    "macro_f1": 0.7350382659085756,
    "negative_adaptation_rate": 0.039634636392096356,
    "f1_delta_vs_proteus_pp": 0.07198983982302787,
    "negative_delta_vs_proteus_pp": -0.08544101884496794,
    "positive_paired_cells": 6,
    "target_pass": false,
    "pre_registered_primary": false
  },
  "dynamic_lr_only": {
    "macro_f1": 0.7347687641596219,
    "negative_adaptation_rate": 0.0398599906775937,
    "f1_delta_vs_proteus_pp": 0.04503966492765965,
    "negative_delta_vs_proteus_pp": -0.06290559029523343,
    "positive_paired_cells": 5,
    "target_pass": false,
    "pre_registered_primary": false
  },
  "bn_interp_only": {
    "macro_f1": 0.7350382659085756,
    "negative_adaptation_rate": 0.039634636392096356,
    "f1_delta_vs_proteus_pp": 0.07198983982302787,
    "negative_delta_vs_proteus_pp": -0.08544101884496794,
    "positive_paired_cells": 6,
    "target_pass": false,
    "pre_registered_primary": false
  },
  "mvi_joint": {
    "macro_f1": 0.7347817067242768,
    "negative_adaptation_rate": 0.03985557982489774,
    "f1_delta_vs_proteus_pp": 0.04633392139314907,
    "negative_delta_vs_proteus_pp": -0.06334667556482931,
    "positive_paired_cells": 5,
    "target_pass": false,
    "pre_registered_primary": true
  },
  "mvi_joint_skip": {
    "macro_f1": 0.6965896969547823,
    "negative_adaptation_rate": 0.0927691195192936,
    "f1_delta_vs_proteus_pp": -3.772867055556295,
    "negative_delta_vs_proteus_pp": 5.228007293874757,
    "positive_paired_cells": 0,
    "target_pass": false,
    "pre_registered_primary": false
  }
}

## 审计恢复说明

首次 postrun 审计的两个记录问题已登记在 `repair_history.md`，原始 scores 备份保存在 `artifacts/scores_pre_repair.json` 和 `artifacts/scores_pre_checkpoint_repair.json`。恢复审计 `artifacts/audit.json` 通过：72 条结果、45 个新 checkpoint、45 次 checkpoint 重载、0 个 GMM 文件、0 次伪标签调用和 15 个配对 stream 单元。恢复后的 checkpoint 重载使用 atol=1e-2，并要求代表 batch argmax agreement≥99%。
