# Peer归因实验结果

GPU0 两个新条件×三 seed完成；上一轮结果只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_only | 89.150% | 88.944% |
| dual_mask_ce | 88.431% | 88.282% |
| mask_peer | 89.477% | 89.346% |
| primary_ce_only_joint | 87.712% | 87.536% |
| primary_ce_peer | 88.497% | 88.359% |

```json
{
  "primary_ce_peer - primary_ce_only_joint": {
    "accuracy_mean_delta_pp": 0.7843137254901933,
    "accuracy_per_seed_pp": [
      0.7843137254901933,
      0.7843137254901933,
      0.7843137254901933
    ],
    "macro_f1_mean_delta_pp": 0.8230712642477359,
    "positive_seed_count": 3,
    "pass": false
  },
  "primary_ce_only_joint - ce_only": {
    "accuracy_mean_delta_pp": -1.4379084967320246,
    "accuracy_per_seed_pp": [
      -1.5686274509803866,
      -1.764705882352946,
      -0.9803921568627416
    ],
    "macro_f1_mean_delta_pp": -1.4073689563885696,
    "positive_seed_count": 0,
    "pass": false
  },
  "primary_ce_peer - ce_only": {
    "accuracy_mean_delta_pp": -0.6535947712418315,
    "accuracy_per_seed_pp": [
      -0.7843137254901933,
      -0.9803921568627527,
      -0.19607843137254832
    ],
    "macro_f1_mean_delta_pp": -0.5842976921408338,
    "positive_seed_count": 0,
    "pass": false
  },
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
  }
}
```

主要比较 primary_ce_peer 与 primary_ce_only_joint；历史 dual_mask_ce/mask_peer/ce_only 仅用于解释。未来/WTT-Time/AWF/TTA关闭。

## 完成后复核与归因结论

6/6 新任务完成，9 个历史参考任务只读复用。95 个冻结文件 hash、60 组指标、30 组 logits、30 组 checkpoint 重载、6 组固定视图诊断、15 组初始化/索引/history 检查通过；preflight 还验证了副视图没有直接 CE 梯度、peer 对称导数、标准 JS 和单网络约束。

`primary_ce_peer` 相对严格匹配的 `primary_ce_only_joint` 平均提升 0.784pp，3/3 seed 提升，Macro-F1 提升 0.823pp。这证明在“主视图 CE、副视图不加 CE、两视图联合前向”的配方内，peer 一致性项有稳定正作用。

但 `primary_ce_only_joint` 只有 87.712%，比历史单视图 CE 低 1.438pp；加入 peer 后 `primary_ce_peer` 仍只有 88.497%，比单视图 CE 低 0.654pp，且 0/3 seed 提升。因此 peer 项的局部归因成功，但整个训练配方仍不具备替代基线的性能。副视图虽无直接 CE 梯度，仍通过联合 BatchNorm 前向影响主分支，故不能把此结果解释成完全隔离的单视图训练。

标准固定视图诊断的三 seed 均值：primary_ce_only_joint masked accuracy 86.536%、预测一致率 97.712%、JS 0.00919；primary_ce_peer 分别为 86.928%、97.582%、JS 0.00823。peer 降低概率差异并略提高 masked accuracy，但没有转化为 clean valid accuracy 的提升。未来/WTT-Time/AWF/TTA 均未访问。
