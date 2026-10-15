# 标签平滑结果

ce-control 与 label-smoothing 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| label_smoothing | 89.150% | 88.994% |

```json
{
  "label_smoothing - ce_control": {
    "accuracy_mean_delta_pp": 3.700743415417189e-15,
    "accuracy_per_seed_pp": [
      0.19607843137255943,
      0.0,
      -0.19607843137254832
    ],
    "macro_f1_mean_delta_pp": 0.0505413250511344,
    "positive_seed_count": 1,
    "pass": false
  }
}
```

epsilon=0.05 仅用于 source CE；未访问未来日期、WTT-Time、AWF 或 TTA。

## 只读审计

`postrun_audit.py` 通过。审计覆盖 17 个冻结文件、12 组指标、12 组 logits/checkpoint、6 组固定 mask、初始化、索引和选模检查；`future_access=false`。审计比较确认 accuracy 平均差为 0，逐 seed 差异为 `+0.196 / 0 / −0.196 pp`，Macro-F1 平均仅 `+0.051 pp`，只有 1/3 seed 提升，预设一致性门槛未通过。

结论：标签平滑 ε=0.05 判定 FAIL，不采用为当前泛化增强模块，也不继续扫描该 ε 配方。Macro-F1 的微小平均改善不足以支持稳定泛化提升。
