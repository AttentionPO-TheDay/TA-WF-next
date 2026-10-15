# SWA结果

三条 CE 轨迹完成；SWA 使用预注册后段七个 checkpoint 的固定平均。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_final | 89.150% | 88.944% |
| swa | 88.235% | 87.980% |

```json
{
  "swa - ce_final": {
    "accuracy_mean_delta_pp": -0.9150326797385625,
    "accuracy_per_seed_pp": [
      -1.17647058823529,
      -1.176470588235301,
      -0.39215686274509665
    ],
    "macro_f1_mean_delta_pp": -0.9640287091267444,
    "positive_seed_count": 0,
    "pass": false
  }
}
```

未访问未来日期、WTT-Time、AWF 或 TTA。

## 复核与结论

`postrun_audit.py` 已通过：冻结 hash、12 组指标、12 组 logits/checkpoint 重载、6 组固定 mask 诊断、3 组初始化、3 组 index stream 和 3 组选模核验均通过；prepared input hash 与祖先冻结输入一致，未访问未来数据。

SWA 相对同轨迹 CE 最佳模型的 accuracy 为 `-0.915pp`，三个 seed 均下降，Macro-F1 为 `-0.964pp`，因此 **FAIL**。当前配方不采用 SWA，也不继续在同一 valid 上扫描平均窗口或学习率。审计限制：本轮未持久化七个单独快照，不能从产物独立重建 SWA 平均；未做 BN 重校准。
