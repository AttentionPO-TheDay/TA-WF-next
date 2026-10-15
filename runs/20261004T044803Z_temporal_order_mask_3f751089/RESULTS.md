# 时间漂移输入增强结果

ce-control 与 temporal-order-mask 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| temporal_order_mask | 89.935% | 89.640% |

```json
{
  "temporal_order_mask - ce_control": {
    "accuracy_mean_delta_pp": 0.7843137254902007,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      1.17647058823529,
      0.39215686274510775
    ],
    "macro_f1_mean_delta_pp": 0.6966925594376505,
    "positive_seed_count": 3,
    "pass": false
  }
}
```

order mask 仅用于 source 训练；valid 使用原始输入；未访问未来日期、WTT-Time、AWF 或 TTA。

## 完成后审计与结论

六项训练均完成 12800 steps、20 次预声明验证选模机会。`postrun_audit.py` 通过：17 个冻结文件哈希、12 组指标、12 组 checkpoint 重载预测，以及 6 组固定 mask、初始化、索引流和选模核验全部通过。

| seed | CE valid accuracy | order-mask valid accuracy | 差异 |
|---|---:|---:|---:|
| 21729 | 89.020% | 89.804% | +0.784pp |
| 23407 | 89.412% | 90.588% | +1.176pp |
| 22026 | 89.020% | 89.412% | +0.392pp |

顺序掩码平均 accuracy 提升 `+0.784pp`，Macro-F1 提升 `+0.697pp`，并且 3/3 seed 都提升。这是目前时间输入增强中最稳定的一轮，但仍未达到预设的平均 accuracy `+1pp` 门槛，因此严格判定 FAIL，暂不冻结为采用方法。固定 span-mask 诊断平均从 CE 的 87.843% 提升到 88.562%，同样只作为辅助证据。

结论仅适用于当前 24-bin 区间均值掩码配方和 TemporalDrift source/valid 开发数据；未访问未来日期、WTT-Time、AWF 或 TTA，不能称为已经验证的跨日期抗漂移方法。
