# Proteus 0.25 倍权重/BN 统计交换诊断结果

结果用于诊断，不把混合 checkpoint 的差值解释为可加的因果贡献。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.856% | 72.579% | 0.000% | 0.000% |
| weights_only | 71.432% | 72.748% | 6.984% | 18.822% |
| bn_stats_only | 68.908% | 70.715% | 6.464% | 9.685% |
| full_strength025 | 73.432% | 74.908% | 4.052% | 19.058% |

## day14

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 86.290% | 0.000% |
| weights_only | 85.151% | 4.626% |
| bn_stats_only | 84.968% | 3.420% |
| full_strength025 | 86.645% | 2.796% |

## day90

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 69.454% | 0.000% |
| weights_only | 70.811% | 6.482% |
| bn_stats_only | 67.441% | 5.605% |
| full_strength025 | 72.151% | 4.231% |

## day270

| 条件 | Macro-F1 | 负适应率 |
|---|---:|---:|
| frozen | 56.824% | 0.000% |
| weights_only | 58.334% | 9.844% |
| bn_stats_only | 54.315% | 10.366% |
| full_strength025 | 61.499% | 5.128% |

## 独立结果复核

`artifacts/diagnostic_audit.json` 通过：从 36 份保存预测重新计算 accuracy、Macro-F1、负适应率、纠错率和变化率，均与 `scores.json` 一致；固定配对行和 9 个 full_strength025 预测与来源 run 精确一致。该诊断仍不能把混合状态差值解释为可加的因果贡献。
