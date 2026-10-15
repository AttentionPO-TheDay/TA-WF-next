# SAM锐度优化结果

GPU0 两个新条件×三 seed完成；CE为历史只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_only | 89.150% | 88.944% |
| two_pass_control | 88.758% | 88.592% |
| sam_ce | 87.516% | 87.155% |

```json
{
  "sam_ce - two_pass_control": {
    "accuracy_mean_delta_pp": -1.2418300653594727,
    "accuracy_per_seed_pp": [
      0.5882352941176561,
      -2.5490196078431393,
      -1.764705882352935
    ],
    "macro_f1_mean_delta_pp": -1.4371576136282038,
    "positive_seed_count": 1,
    "pass": false
  },
  "two_pass_control - ce_only": {
    "accuracy_mean_delta_pp": -0.39215686274509665,
    "accuracy_per_seed_pp": [
      -0.19607843137254832,
      0.19607843137254832,
      -1.17647058823529
    ],
    "macro_f1_mean_delta_pp": -0.3517615282321159,
    "positive_seed_count": 1,
    "pass": false
  },
  "sam_ce - ce_only": {
    "accuracy_mean_delta_pp": -1.6339869281045694,
    "accuracy_per_seed_pp": [
      0.39215686274510775,
      -2.352941176470591,
      -2.941176470588225
    ],
    "macro_f1_mean_delta_pp": -1.7889191418603196,
    "positive_seed_count": 1,
    "pass": false
  }
}
```

SAM只改变第一遍梯度后的参数扰动；两次前向控制用于分离计算流程影响。未来/WTT-Time/AWF/TTA关闭。

## 复核

`postrun_audit.py` 已通过（审计脚本 SHA-256：`e14e3eaed931d8dd87dc3dcbcd67292ce7e34121671c1122b54b6608b84d4bd5`）：

- 41/41 冻结文件 hash 一致；6/6 新训练任务完成。
- 36 组指标、18 组 logits、18 组 checkpoint 重载、9 组初始化/索引流/选模核验通过。
- 6 组 SAM 扰动历史、BN 两遍规则和参数精确恢复检查通过。
- 未访问未来日期、WTT-Time、AWF 或 TTA；未重跑历史 CE 全训练。

## 结论与后续

在当前 wide+BN 配方、`rho=0.05` 和 BN 处理规则下，SAM 相对严格两遍控制的 valid accuracy 为 `-1.242pp`，仅 1/3 seed 提升；相对历史单视图 CE 为 `-1.634pp`。因此本实验为 **FAIL**，不采用 SAM，也不继续扫描其 `rho`、教师/学生或蒸馏超参数。后续泛化研究应转向尚未验证的独立候选（例如 SWA 或时间结构模块），并重新建立有界对照。
