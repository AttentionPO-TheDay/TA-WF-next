# 无标签更新接受与回退实验结果

全部 3 个训练 seed × 3 个开发日期完成；support 和 query 精确隔离，目标标签只在预测固定后用于评分。global 条件逐单元精确复现上一轮 full_global 预测。

| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 | 纠错率均值 | 接受步数均值 | 特征相对改变量 |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% | 0.00 | 0.000000 |
| global | 70.973% | 72.714% | 1.032% | 3.622% | 10.00 | 0.013323 |
| stepwise_rollback | 70.985% | 72.733% | 0.684% | 2.841% | 5.89 | 0.010203 |
| episode_rollback | 70.973% | 72.714% | 1.032% | 3.622% | 10.00 | 0.013323 |

裁决 JSON：
```json
{
  "primary_stepwise": {
    "macro_f1_minus_global_mean_pp": 0.012039939705732472,
    "negative_adaptation_minus_global_mean_pp": -0.3484408820711428,
    "cells_with_lower_negative_adaptation": 9,
    "pass": true
  },
  "secondary_episode": {
    "macro_f1_minus_global_mean_pp": 0.0,
    "negative_adaptation_minus_global_mean_pp": 0.0,
    "cells_with_lower_negative_adaptation": 0,
    "pass": false
  },
  "pass": true
}
```

逐单元结果及接受轨迹见 `artifacts/scores.json`；完整 logits、适配器 checkpoint、split、冻结 hash 和独立审计均已保留。

本轮是 TemporalDrift 已观察数据上的开发筛查；没有重新训练生成器或 backbone，也未访问 WTT-Time、AWF 或其他漂移因素数据。
