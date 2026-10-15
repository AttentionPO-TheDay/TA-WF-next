# 生成器固定统计与学习特征分离适应结果

全部 3 个训练 seed × 3 个开发日期完成；query 与无标签 adaptation reference 按精确输入指纹隔离，目标标签仅用于预测后评分。

| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 | 特征相对改变量 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000000 |
| full_global | 70.973% | 72.714% | 1.032% | 0.013323 |
| full_masked_learned_global | 70.922% | 72.649% | 0.536% | 0.011689 |
| learned_global | 70.925% | 72.650% | 0.477% | 0.010493 |
| learned_drift_selective | 70.887% | 72.612% | 0.240% | 0.007648 |
| learned_random_selective | 70.894% | 72.613% | 0.242% | 0.007374 |

裁决 JSON：
```json
{
  "learned_global_minus_full_global_macro_f1_mean_pp": -0.04874955631247735,
  "selective_minus_global_macro_f1_mean_pp": -0.037264902754574614,
  "selective_minus_random_macro_f1_mean_pp": -0.006281266709196043,
  "selective_minus_global_negative_rate_pp": -0.2370101395045429,
  "positive_cells_vs_global": 2,
  "positive_cells_vs_random": 4,
  "pass": false
}
```

逐单元结果见 artifacts/scores.json；完整logits、门控、适配器checkpoint、逐行split、源统计与冻结hash均保留。

full_masked_learned_global 是与完整110维适配器同结构、同初始化的输出屏蔽对照，用于区分固定统计保护与适配器维度变化。学习分支的三个条件使用相同80维结构和初始化。

相对改变量在support上用完整110维输入与残差的L2比计算；只记录幅度，尚未做幅度匹配，因此安全性差异不能单独归因于选区质量。

本轮仅为TemporalDrift开发筛查，未来日期已观察；固定统计不是已证明的稳定因素，本轮未重新训练生成器，未访问WTT-Time、AWF、Network/Behavior/Version目标。
