# 生成器局部表示选择性适应结果

全部 3 个训练 seed × 3 个开发日期完成；query 与无标签 adaptation reference 按精确输入指纹隔离，目标标签仅用于预测后评分。

| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 |
|---|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% |
| global | 70.973% | 72.714% | 1.032% |
| drift_selective | 70.949% | 72.679% | 0.615% |
| random_selective | 70.909% | 72.640% | 0.642% |

| 日期 | frozen Macro-F1 | global Macro-F1 | drift-selective Macro-F1 | random-selective Macro-F1 |
|---|---:|---:|---:|---:|
| Day14 | 86.289% | 86.490% | 86.462% | 86.444% |
| Day90 | 69.459% | 69.477% | 69.499% | 69.455% |
| Day270 | 56.823% | 56.953% | 56.886% | 56.827% |

裁决 JSON：
```json
{
  "selective_minus_global_macro_f1_mean_pp": -0.024145318266410865,
  "selective_minus_random_macro_f1_mean_pp": 0.04031809563262989,
  "selective_minus_global_negative_rate_pp": -0.4168575523681117,
  "positive_cells_vs_global": 5,
  "positive_cells_vs_random": 7,
  "pass": false
}
```

逐单元结果见 artifacts/scores.json；逐条预测、适配器 checkpoint、输入 hash 和冻结文件见 artifacts/。

本轮仅为 TemporalDrift 开发证据，未访问 WTT-Time、AWF 或外部未来测试。

解释：漂移选择相对全局适应的平均 Macro-F1 为 -0.024pp，9 个单元中 5 个提升；相对随机选择为 +0.040pp，9 个单元中 7 个提升。漂移选择平均负适应率为 0.615%，全局适应为 1.032%，降低 0.417pp。因未同时满足预注册的性能门槛，本轮不能支持“选择性适应取得更好识别效果”的结论；它支持一个较窄的观察，即在当前零初始化局部适配器和熵更新配方下，漂移分数选择减少了负适应，但性能收益尚未超过全局适应。

首次执行的随机门控 dtype 错误及同配置复测记录见 `repair_history/README.md`；失败 traceback 保留用于审计。
