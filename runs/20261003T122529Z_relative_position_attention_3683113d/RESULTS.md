# 相对位置注意力结果

ce-control 与 relative-position 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| relative_position | 6.078% | 3.582% |

```json
{
  "relative_position - ce_control": {
    "accuracy_mean_delta_pp": -83.07189542483661,
    "accuracy_per_seed_pp": [
      -83.33333333333333,
      -83.33333333333334,
      -82.54901960784314
    ],
    "macro_f1_mean_delta_pp": -85.36149440239733,
    "positive_seed_count": 0,
    "pass": false
  }
}
```

固定相对距离 bias 未训练；未访问未来日期、WTT-Time、AWF 或 TTA。

## 复核与结论

`postrun_audit.py` 已通过：17 个冻结文件、12 组指标、12 组 logits/checkpoint 重载、6 组固定 mask、6 组初始化、6 组 index stream 和 6 组选模核验通过，未访问未来数据。

相对位置偏置导致 valid accuracy 仅 `6.078%`，相对严格 CE 控制下降 `83.072pp`，三个 seed 均严重下降。该实现为明确 **FAIL**；停止相对位置偏置方向，不继续扫描 bias 强度或位置编码变体。结果提示当前 Transformer 对该 additive mask 的数值/注意力约束不匹配，不能据此否定所有位置结构方法。
