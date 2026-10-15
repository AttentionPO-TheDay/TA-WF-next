# 监督对比结果

ce-control 与 supcon 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| supcon | 88.105% | 87.882% |

```json
{
  "supcon - ce_control": {
    "accuracy_mean_delta_pp": -1.0457516339869244,
    "accuracy_per_seed_pp": [
      0.39215686274510775,
      -2.352941176470591,
      -1.17647058823529
    ],
    "macro_f1_mean_delta_pp": -1.0617378264437132,
    "positive_seed_count": 1,
    "pass": false
  }
}
```

SupCon 仅训练期使用投影头；未访问未来日期、WTT-Time、AWF 或 TTA。

## 复核与结论

`postrun_audit.py` 已通过：17 个冻结文件、12 组指标、12 组 logits/checkpoint 重载、6 组固定 mask、6 组初始化、6 组 index stream 和 6 组选模核验通过。

SupCon 相对严格 `ce_control` 的 valid accuracy 为 `-1.046pp`，仅 1/3 seed 提升；Macro-F1 为 `-1.062pp`。因此本实验为 **FAIL**，不采用当前 `weight=0.1, temperature=0.1` 的监督对比方案，也不继续在同一 valid 上扫描权重或温度。
