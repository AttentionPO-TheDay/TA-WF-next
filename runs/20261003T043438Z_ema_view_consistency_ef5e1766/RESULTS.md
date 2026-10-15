# EMA 视图一致性结果

GPU0 三条件×三 seed 正式训练完成。

| 条件 | valid accuracy | valid Macro-F1 | ≥90% |
|---|---:|---:|---|
| ce_only | 89.150% | 88.944% | False |
| ema_same_view | 88.431% | 88.138% | False |
| ema_cross_view | 88.562% | 88.456% | False |

```json
{
  "ema_same_view - ce_only": {
    "accuracy_mean_delta_pp": -0.7189542483660143,
    "accuracy_per_seed_pp": [
      -0.39215686274509665,
      -1.764705882352946,
      0.0
    ],
    "macro_f1_mean_delta_pp": -0.8055270800368867,
    "positive_seed_count": 0,
    "pass": false
  },
  "ema_cross_view - ce_only": {
    "accuracy_mean_delta_pp": -0.5882352941176487,
    "accuracy_per_seed_pp": [
      -0.39215686274509665,
      -1.3725490196078494,
      0.0
    ],
    "macro_f1_mean_delta_pp": -0.487208869561806,
    "positive_seed_count": 0,
    "pass": false
  }
}
```

本轮无固定教师 checkpoint、无教师 logits；EMA 副本仅由各学生在线更新。未来/WTT-Time/AWF/TTA关闭。

## 完成后复核（2026-10-03）

9/9 项任务完成，用时约 33.26 分钟。新增 `postrun_audit.py` 只用于完成后复核，未修改原冻结配置、训练代码或原冻结 hash 清单，也未训练或新增数据角色。17 个冻结文件 hash、36 组独立 accuracy/Macro-F1、18 组保存 logits、18 组 GPU checkpoint 重载推理通过；9 组初始化、索引流和 history/最早最佳选模一致。CE 重现上一轮相同 seed 的预测；EMA 副本构造和 eval 前向未消耗学生 dropout 使用的 CUDA RNG。EMA 参数更新公式、无梯度设置及 BN buffer 同步规则已验证。

## 结论和解释边界

同视图 EMA 相对 CE 平均 accuracy −0.719pp、Macro-F1 −0.806pp；跨视图 EMA 为 −0.588pp、−0.487pp。两条件均为 2/3 seed 下降、1/3 持平，无正向 seed，均未达冻结候选门槛。保留 CE 89.150% 基线，不采用本轮 EMA 一致性配方；这不能证明所有蒸馏机制都不适合该模型，也未定位唯一失败原因。

本轮实际配置没有置信度门控，使用从第 1 步开始的权重线性 ramp，因而不等同于此前聊天中提出的置信度门控 EMA 方案。EMA 网络处于 eval，主网络处于 train；同视图仅指输入 span-mask 相同，dropout/BN 工作模式并不相同。最终指标来自主网络，未对 EMA 权重推理评分，且未保留完整 EMA 轨迹，不能据此判断 EMA 权重用于推理的效果。

此前称该方法“没有教师模型”不够准确：它没有固定预训练教师，但仍有无梯度的 EMA 参考网络，属于自蒸馏/一致性训练；并未完全取消双网络结构。若后续要求真正单网络且无教师角色，需另立方法与有界实验，不能将本轮改称单网络实验。当前结果只来自已观察 source/valid，未访问未来日期、WTT-Time、AWF 或 TTA，不能视作抗时间漂移的证据。
