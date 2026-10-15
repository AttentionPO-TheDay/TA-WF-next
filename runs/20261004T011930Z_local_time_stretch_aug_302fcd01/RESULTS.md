# 时间漂移输入增强结果

ce-control 与 local-time-stretch 各三 seed 完成；历史 CE 为只读参考。

| 条件 | valid accuracy | valid Macro-F1 |
|---|---:|---:|
| ce_control | 89.150% | 88.944% |
| local_time_stretch | 89.673% | 89.381% |

```json
{
  "local_time_stretch - ce_control": {
    "accuracy_mean_delta_pp": 0.5228758169934696,
    "accuracy_per_seed_pp": [
      0.39215686274510775,
      -0.19607843137254832,
      1.3725490196078494
    ],
    "macro_f1_mean_delta_pp": 0.43774671225651857,
    "positive_seed_count": 2,
    "pass": false
  }
}
```

stretch 仅用于 source 训练；valid 使用原始输入；未访问未来日期、WTT-Time、AWF 或 TTA。

## 完成后审计与结论

全部六项训练均完成 12800 steps 和 20 次预声明验证选模机会；总耗时 2110.435 秒（约 35.2 分钟）。独立 postrun_audit 通过：17 个冻结文件哈希、12 组指标、12 组 checkpoint 重载预测及 6 组固定 mask、初始化、索引流、选模核验全部通过。审计在空闲 GPU0 执行，无新增训练。

| seed | CE valid accuracy | local stretch valid accuracy | 差异 |
|---|---:|---:|---:|
| 21729 | 89.020% | 89.412% | +0.392pp |
| 23407 | 89.412% | 89.216% | −0.196pp |
| 22026 | 89.020% | 90.392% | +1.373pp |

判定：FAIL，不采用当前配方。平均 accuracy +0.523pp、Macro-F1 +0.438pp，为小幅正向开发信号；但平均 accuracy 增益低于预设 +1pp，且仅 2/3 seed 正向，未满足 3/3 一致性门槛。不能以单个 seed 达到 90.392% 代替三 seed 平均结果。

预声明固定 span-mask 诊断：CE 87.843% / local stretch 88.562%，平均 +0.719pp；逐 seed 为 +0.588/−0.784/+2.353pp，也仅 2/3 正向。该诊断不参与选模，不能替代主要门槛。

限制：只使用已观察的 source/valid；未评分任何未来日期，不能称为跨日期抗漂移成功。实现是局部坐标插值加全轴每通道总量重标定，不能把增强作用唯一归因为局部加载节奏不变性，也不能把本轮阴性推广为所有时间增强无效。不因小幅均值改善默认继续在同一 valid 扫描参数。
