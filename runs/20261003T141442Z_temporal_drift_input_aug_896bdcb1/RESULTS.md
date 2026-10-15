# 时间漂移输入增强结果

ce-control 与 temporal-origin-jitter 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| temporal_origin_jitter | 89.346% | 89.158% |

```json
{
  "temporal_origin_jitter - ce_control": {
    "accuracy_mean_delta_pp": 0.19607843137255201,
    "accuracy_per_seed_pp": [
      0.7843137254902044,
      -0.9803921568627527,
      0.7843137254902044
    ],
    "macro_f1_mean_delta_pp": 0.21408330231859432,
    "positive_seed_count": 2,
    "pass": false
  }
}
```

jitter 仅用于 source 训练；valid 使用原始输入；未访问未来日期、WTT-Time、AWF 或 TTA。

## 完成后审计与判定

2026-10-04 完成只读审计：17 个冻结文件哈希、12 组 source/valid 指标、12 组 checkpoint 重载预测、6 组固定 mask、初始化、索引流和选模核验全部通过。六项训练均完成 12800 steps、20 次预声明验证选模机会。审计使用空闲 GPU0，不进行新训练。

| seed | CE valid accuracy | jitter valid accuracy | 差异 |
|---|---:|---:|---:|
| 21729 | 89.020% | 89.804% | +0.784pp |
| 23407 | 89.412% | 88.431% | −0.980pp |
| 22026 | 89.020% | 89.804% | +0.784pp |

按预设门槛判定 FAIL，不采用当前时间原点抖动配方。平均 accuracy +0.196pp、Macro-F1 +0.214pp，仅 2/3 seed accuracy 正向；既未达到 +1pp 的幅度门槛，也未达到 3/3 seed 一致性门槛。固定 span-mask 诊断准确率 CE 87.843% / jitter 87.712%，平均 −0.131pp，也不支持遮挡鲁棒性稳定提升。

限制：本轮是已观察 source/valid 上的机制开发，未评分未来日期；时间轴平移只模拟一个可能的漂移因素，不能证明或否定跨日期抗漂移能力，更不能据此否定所有输入级增强。该平移越界丢弃计数、补零，不保证总计数守恒（启动前已更正表述）。不因本轮小幅均值提升默认继续追调幅度或概率。
