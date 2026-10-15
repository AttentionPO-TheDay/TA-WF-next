# Proteus 机制忠实移植 baseline 结果

本 run 复现官方 Proteus 的损失项、动态带宽 MMD、GMM 伪标签和 100 epoch 更新，但底座是当前 wide+BN，适应使用独立无标签 support，query 标签只在最终预测后评分。因此结果是机制移植 baseline，不是官方 RF 端到端复现。

| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 | 纠错率均值 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| global_entropy_adapter | 70.973% | 72.714% | 1.032% | 3.622% |
| stepwise_rollback | 70.985% | 72.733% | 0.684% | 2.841% |
| proteus_port | 70.793% | 72.251% | 8.512% | 22.143% |

裁决与对照：
```json
{
  "proteus_minus_frozen_macro_f1_mean_pp": -0.06416422524259113,
  "proteus_minus_global_macro_f1_mean_pp": -0.18077242959161355,
  "proteus_minus_stepwise_macro_f1_mean_pp": -0.19281236929734602,
  "proteus_minus_frozen_negative_rate_pp": 8.51212938243226,
  "proteus_minus_global_negative_rate_pp": 7.4801443150710965,
  "proteus_mean_pseudo_coverage": 0.8845486111111112
}
```

全部9个Proteus任务的无标签损失、GMM覆盖率、query预测和checkpoint已保存；未按query标签选epoch或阈值。执行后补充了 Proteus 派生指标的逐行审计复算，复核通过；修订原因见 `repair_history.md`。
