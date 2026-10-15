# 相对时间结构结果

ce-control 与 relative-time 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 87.451% | 87.134% |
| relative_time | 87.190% | 86.919% |

```json
{
  "relative_time - ce_control": {
    "accuracy_mean_delta_pp": -0.2614379084967311,
    "accuracy_per_seed_pp": [
      0.39215686274509665,
      -1.9607843137254832,
      0.7843137254901933
    ],
    "macro_f1_mean_delta_pp": -0.21536262020690847,
    "positive_seed_count": 2,
    "pass": false
  }
}
```

四个时间特征只由当前输入的 signed timestamps 计算；未访问未来日期、WTT-Time、AWF 或 TTA。

## 复核与结论

修订后的 `postrun_audit.py` 已通过：17 个冻结文件、12 组指标、12 组 logits/checkpoint 重载、6 组固定 mask 诊断、6 组初始化、6 组 index stream、6 组选模及 6 组时间特征参考核验均通过；prepared input hash 与祖先冻结输入一致。初版审计断言失败仅由新增投影参数的 key 集合比较过严造成，已在 PLAN.md 登记。

`relative_time` 相对严格 `ce_control` 的 valid accuracy 为 `-0.261pp`，逐 seed 为 `+0.392/-1.961/+0.784pp`，仅 2/3 seed 提升；Macro-F1 为 `-0.215pp`。因此本实验为 **FAIL**，不采用当前四统计量时间结构，也不继续在同一 valid 上扫描时间特征或投影配置。
