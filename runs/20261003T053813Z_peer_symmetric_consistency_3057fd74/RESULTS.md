# 单网络双视图互蒸馏结果

GPU0 四个新条件×三 seed 完成；ce_only 为历史只读参考。

| 条件 | valid accuracy | valid Macro-F1 | ≥90% |
|---|---:|---:|---|
| ce_only | 89.150% | 88.944% | False |
| dual_mask_ce | 88.431% | 88.282% | False |
| mask_peer | 89.477% | 89.346% | False |
| dual_cross_ce | 87.516% | 87.258% | False |
| cross_peer | 87.778% | 87.545% | False |

```json
{
  "mask_peer - dual_mask_ce": {
    "accuracy_mean_delta_pp": 1.045751633986932,
    "accuracy_per_seed_pp": [
      0.9803921568627527,
      0.9803921568627527,
      1.17647058823529
    ],
    "macro_f1_mean_delta_pp": 1.0642063093043608,
    "positive_seed_count": 3,
    "pass": true
  },
  "cross_peer - dual_cross_ce": {
    "accuracy_mean_delta_pp": 0.26143790849672743,
    "accuracy_per_seed_pp": [
      -0.7843137254902044,
      0.19607843137254832,
      1.3725490196078383
    ],
    "macro_f1_mean_delta_pp": 0.2867103847495889,
    "positive_seed_count": 2,
    "pass": false
  },
  "dual_mask_ce - ce_only": {
    "accuracy_mean_delta_pp": -0.7189542483660143,
    "accuracy_per_seed_pp": [
      -0.19607843137254832,
      -0.7843137254902044,
      -1.17647058823529
    ],
    "macro_f1_mean_delta_pp": -0.6619451717490972,
    "positive_seed_count": 0,
    "pass": false
  },
  "dual_cross_ce - ce_only": {
    "accuracy_mean_delta_pp": -1.6339869281045694,
    "accuracy_per_seed_pp": [
      0.19607843137255943,
      -2.7450980392156876,
      -2.35294117647058
    ],
    "macro_f1_mean_delta_pp": -1.6852646264410927,
    "positive_seed_count": 1,
    "pass": false
  },
  "mask_peer - ce_only": {
    "accuracy_mean_delta_pp": 0.32679738562091754,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      0.19607843137254832,
      0.0
    ],
    "macro_f1_mean_delta_pp": 0.40226113755526355,
    "positive_seed_count": 2,
    "pass": false
  },
  "cross_peer - ce_only": {
    "accuracy_mean_delta_pp": -1.3725490196078418,
    "accuracy_per_seed_pp": [
      -0.588235294117645,
      -2.5490196078431393,
      -0.9803921568627416
    ],
    "macro_f1_mean_delta_pp": -1.398554241691504,
    "positive_seed_count": 0,
    "pass": false
  }
}
```

peer 条件只有一个可训练网络；两视图共享参数，彼此 stop-gradient 概率互为目标。dual_view_ce 用于分离双视图 CE 的作用。未来/WTT-Time/AWF/TTA关闭。

## 完成后复核

12/12 新任务完成。17 个冻结文件 hash、48 组指标、24 组 logits、24 组 checkpoint 重载、12 组固定视图诊断、12 组 history/选模、初始化和索引流检查通过；历史 CE 的 3 个 seed 预测重现，确认每个 peer 条件只有一个网络且未加载固定教师或 EMA。固定视图诊断只读执行，不参与选模、阈值或损失调整。

## 结论边界

`mask_peer` 相对匹配的 `dual_mask_ce` 平均提升 1.046pp，3/3 seed 提升，Macro-F1 提升 1.064pp，说明在本轮双 mask CE 对照下，对称互蒸馏确实恢复了双视图训练损失。但相对历史单视图 CE 仅 +0.327pp，2/3 seed 提升，未达到预设 +1pp 且 3/3 seed 的采用门槛，因此暂不替换 89.150% CE 基线。

`cross_peer` 相对 `dual_cross_ce` 仅 +0.261pp、2/3 seed 提升；clean/mask 跨视图本身使双视图 CE 降至 87.516%，peer 不能恢复到单视图 CE 水平。当前最有价值的局部证据是 mask-peer 对匹配双视图 CE 的稳定改善，不能表述为整体分类性能稳定提升或时间漂移缓解。
