# 轻量超参数敏感性结果

63个新适应单元及18个历史控制单元共81条指标已全query重算；63个新checkpoint独立重载。所有结果是已观察TemporalDrift开发证据。

| 配方 | 配置 | Macro-F1 (%) | 相对本配方控制 ΔF1 (pp) | NAR (%) | Recovery (%) | 配对正向/9 |
|---|---|---:|---:|---:|---:|---:|
| no_pseudo_fixed | control_no_pseudo_fixed: 原800步/lr0.00025 | 74.755 | +0.000 | 3.708 | 17.786 | — |
| no_pseudo_fixed | no_pseudo_lr_half_800: {'condition': 'no_pseudo_fixed', 'learning_rate': 0.000125, 'candidate_updates': 800} | 74.830 | +0.074 | 3.483 | 17.207 | 5 |
| no_pseudo_fixed | no_pseudo_lr_double_800: {'condition': 'no_pseudo_fixed', 'learning_rate': 0.0005, 'candidate_updates': 800} | 73.417 | -1.338 | 5.911 | 18.869 | 0 |
| no_pseudo_fixed | no_pseudo_steps_half_400: {'condition': 'no_pseudo_fixed', 'learning_rate': 0.00025, 'candidate_updates': 400} | 74.681 | -0.074 | 3.634 | 17.197 | 1 |
| tent_entropy_gate | control_tent_entropy_gate: 原800步/lr0.00025/门控[0.2,0.4] | 72.969 | +0.000 | 2.101 | 7.629 | — |
| tent_entropy_gate | gate_lr_half_800: {'condition': 'tent_entropy_gate', 'learning_rate': 0.000125, 'candidate_updates': 800, 'gate_nats': [0.2, 0.4]} | 72.989 | +0.020 | 1.461 | 5.735 | 5 |
| tent_entropy_gate | gate_lr_double_800: {'condition': 'tent_entropy_gate', 'learning_rate': 0.0005, 'candidate_updates': 800, 'gate_nats': [0.2, 0.4]} | 72.833 | -0.137 | 2.813 | 9.338 | 2 |
| tent_entropy_gate | gate_narrow_800: {'condition': 'tent_entropy_gate', 'learning_rate': 0.00025, 'candidate_updates': 800, 'gate_nats': [0.25, 0.35]} | 72.977 | +0.007 | 1.666 | 6.220 | 5 |
| tent_entropy_gate | gate_wide_800: {'condition': 'tent_entropy_gate', 'learning_rate': 0.00025, 'candidate_updates': 800, 'gate_nats': [0.1, 0.5]} | 72.990 | +0.020 | 2.548 | 9.004 | 5 |

逐 seed×日期预测、配对指标、成本与动态日志在 artifacts。两组配方分别对照，不跨配方把一个超参数变化归因为方法优劣。TENT门控样本数会随区间变化，属于本敏感性变量。三日期和三训练seed共享既有开发query，不作为独立确认或显著性样本。外部WTT/AWF未重新评分。
