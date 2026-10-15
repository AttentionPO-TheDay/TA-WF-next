# 时间顺序掩码组件结果

CE 与四种顺序掩码组件各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| hard_scalar | 89.935% | 89.640% |
| soft_scalar | 89.020% | 88.887% |
| hard_shape | 88.758% | 88.545% |
| soft_shape | 88.235% | 88.151% |

```json
{
  "hard_scalar": {
    "accuracy_mean_delta_pp": 0.7843137254902007,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      1.17647058823529,
      0.39215686274510775
    ],
    "macro_f1_mean_delta_pp": 0.6966925594376505,
    "positive_seed_count": 3,
    "pass": false
  },
  "soft_scalar": {
    "accuracy_mean_delta_pp": -0.13071895424836186,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      0.588235294117645,
      -1.764705882352935
    ],
    "macro_f1_mean_delta_pp": -0.05625819351309606,
    "positive_seed_count": 2,
    "pass": false
  },
  "hard_shape": {
    "accuracy_mean_delta_pp": -0.39215686274509665,
    "accuracy_per_seed_pp": [
      0.0,
      -1.3725490196078494,
      0.19607843137255943
    ],
    "macro_f1_mean_delta_pp": -0.39875338894946877,
    "positive_seed_count": 1,
    "pass": false
  },
  "soft_shape": {
    "accuracy_mean_delta_pp": -0.9150326797385588,
    "accuracy_per_seed_pp": [
      -1.9607843137254832,
      1.3725490196078383,
      -2.1568627450980316
    ],
    "macro_f1_mean_delta_pp": -0.7921834882619216,
    "positive_seed_count": 1,
    "pass": false
  }
}
```

组件掩码仅用于 source 训练；valid 使用原始输入；未访问未来日期、WTT-Time、AWF 或 TTA。

## 完成后审计与结论

15 项训练均完成 12800 steps、20 次预声明验证选模机会。`postrun_audit.py` 通过：17 个冻结文件哈希、30 组指标、30 组 checkpoint 重载预测，以及 15 组固定 mask、初始化、索引流和选模核验全部通过。

| 组件 | valid accuracy | 相对 CE | Macro-F1 | 正向 seed |
|---|---:|---:|---:|---:|
| hard scalar | 89.935% | +0.784pp | 89.640% | 3/3 |
| soft scalar | 89.020% | −0.131pp | 88.887% | 2/3 |
| hard shape | 88.758% | −0.392pp | 88.545% | 1/3 |
| soft shape | 88.235% | −0.915pp | 88.151% | 1/3 |

组件归因结论：收益来自**完整的标量均值替换**，而不是更温和的混合强度或保留 bin 内细粒度形状。soft scalar 已失去平均收益；两种 shape 模板均下降，说明当前模型更需要消除局部时间细节，而不是保留该细节。hard scalar 仍未达到预设平均 accuracy +1pp 门槛，因此严格判定 FAIL；它是当前唯一保留的开发候选，不进入外部评价或继续无界调参。

限制：本轮只使用 TemporalDrift source/valid 开发数据，未访问未来日期、WTT-Time、AWF 或 TTA；组件结论只适用于当前 24-bin、p=0.75 和三 seed 配方。
