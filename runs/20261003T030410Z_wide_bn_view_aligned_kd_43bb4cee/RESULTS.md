# 视图对齐蒸馏结果

GPU0 四条件×三 seed 正式训练完成。

| 条件 | valid accuracy | valid Macro-F1 | ≥90% |
|---|---:|---:|---|
| ce_only | 89.150% | 88.944% | False |
| matched_kd | 89.412% | 89.232% | False |
| clean_kd | 88.431% | 88.269% | False |
| gated_matched_kd | 87.843% | 87.645% | False |

```json
{
  "matched_kd - ce_only": {
    "accuracy_mean_delta_pp": 0.26143790849673476,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      -0.7843137254902044,
      0.7843137254902044
    ],
    "macro_f1_mean_delta_pp": 0.2883898962330286,
    "positive_seed_count": 2,
    "pass": false
  },
  "clean_kd - ce_only": {
    "accuracy_mean_delta_pp": -0.7189542483660105,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      -2.1568627450980427,
      -0.7843137254901933
    ],
    "macro_f1_mean_delta_pp": -0.6747591551513128,
    "positive_seed_count": 1,
    "pass": false
  },
  "gated_matched_kd - ce_only": {
    "accuracy_mean_delta_pp": -1.307189542483659,
    "accuracy_per_seed_pp": [
      -0.588235294117645,
      -3.3333333333333326,
      0.0
    ],
    "macro_f1_mean_delta_pp": -1.2988536517948255,
    "positive_seed_count": 0,
    "pass": false
  }
}
```

教师目标为逐教师温度概率平均；matched 与 clean 条件分别固定输入视图，gated 只按教师置信度和投票分歧门控。valid 仅用于预定选模与评价；未来/WTT-Time/AWF/TTA关闭。

## 复核与限制

12/12 任务完成。26 个冻结文件 hash、48 组指标、24 组 logits、12 组 history/选模、12 组初始化和 12 组索引流检查通过；未访问未来数据。启动阶段曾因教师构造影响随机状态而中止，修复后重置学生初始化并重新完成全批次，最终各条件初始化和索引流均与 CE 对照一致。

`gated_matched_kd` 的实际门控覆盖率仅为 1–3 / 819200 个 source batch 样本（约 0.0001–0.0004%），因此本条件基本等价于 CE，不能解释为充分测试了选择性蒸馏。
