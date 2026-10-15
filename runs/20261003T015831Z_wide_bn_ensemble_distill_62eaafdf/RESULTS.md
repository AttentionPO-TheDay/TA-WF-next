# 集成蒸馏结果

GPU0 三条件×三 seed 正式训练完成。

| 条件 | valid accuracy | valid Macro-F1 | ≥90% |
|---|---:|---:|---|
| ce_only | 89.150% | 88.944% | False |
| ensemble_kd | 89.412% | 89.214% | False |
| shuffled_kd | 65.359% | 64.045% | False |

```json
{
  "ensemble_kd - ce_only": {
    "accuracy_mean_delta_pp": 0.26143790849673476,
    "accuracy_per_seed_pp": [
      1.764705882352946,
      0.0,
      -0.9803921568627416
    ],
    "macro_f1_mean_delta_pp": 0.26992796600639657,
    "pass": false
  },
  "shuffled_kd - ce_only": {
    "accuracy_mean_delta_pp": -23.79084967320261,
    "accuracy_per_seed_pp": [
      -22.941176470588232,
      -21.372549019607845,
      -27.058823529411757
    ],
    "macro_f1_mean_delta_pp": -24.899057006900147,
    "pass": false
  }
}
```

教师仅在 source 训练行提供固定软目标；valid 只用于预定选模与评价。shuffled_kd 只打乱教师行对应关系，用于区分样本知识和软目标正则化。未来/WTT-Time/AWF/TTA 关闭。

## 复核

9/9 任务完成；18 组 source/valid checkpoint 重载推理、36 组 best/last 预测指标、9 组初始化状态、9 条索引流和 9 组选模步数均通过复核。三份教师 logits 的 hash 与历史 wide+BN run 对应文件一致，CE 学生的预测类别与历史 wide+BN 基线一致。未来日期、WTT-Time、AWF 和 TTA 未访问。

## 解释

真实集成软目标只有 +0.261pp 的平均 accuracy，且只有 1/3 seed 提升、1 个持平、1 个下降，因此不采用为当前开发候选。打乱教师行后性能下降约 23.791pp，说明逐样本对应关系确实是必要的；但这不能把小幅真实蒸馏增益称为稳定的知识迁移。该结果只支持“固定集成软目标在本配方下未形成稳定单模型提升”，不支持抗时间漂移或外部泛化结论。

## 逐 seed 结果

|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|last valid accuracy|
|---|---:|---:|---:|---:|---:|---:|
|ce_only|21729|10400|100.000%|89.020%|88.822%|87.059%|
|ensemble_kd|21729|12320|100.000%|90.784%|90.539%|89.020%|
|shuffled_kd|21729|7040|76.209%|66.078%|64.800%|46.471%|
|ce_only|23407|12320|100.000%|89.412%|89.239%|89.412%|
|ensemble_kd|23407|11360|99.993%|89.412%|89.168%|88.431%|
|shuffled_kd|23407|6560|73.268%|68.039%|66.545%|55.686%|
|ce_only|22026|7520|99.346%|89.020%|88.770%|88.627%|
|ensemble_kd|22026|9920|99.980%|88.039%|87.934%|87.647%|
|shuffled_kd|22026|7520|68.510%|61.961%|60.789%|54.510%|

## 方法与执行记录补充

训练批次耗时 29.18 分钟。训练前 frozen manifest 的 20 项 hash 复核通过；完成后的审计脚本为新增只读复核代码，未改动训练代码、config、PLAN 或 freeze manifest。代码/配置/输入版本以 `artifacts/freeze.json` 为准，复核明细见 `artifacts/postrun_audit.json`。

此前“CUDA不可用”判断来自 sandbox 对宿主 GPU 设备的访问限制；宿主机 GPU0 RTX 4090 实际可用，随后在宿主权限下完成全批次。首次启动在 optimizer step 前因检查被屏蔽 packet 分支梯度的断言过严停止；修正为检查启用的生成器 stage/output 权重并重新冻结后，从随机初始化重启，没有使用失败首批作为结果。此前失败首批日志/部分产物未保留，是记录保留上的缺口；本补充如实登记，不影响最终9任务的已保存产物核验。

实际蒸馏目标为三教师 **raw logits算术平均** 后 temperature=2 的概率，不是历史91.569%的多数投票规则；历史raw-logit平均的valid accuracy为90.588%。因此不能称已将91.569%多数投票模型原样压缩。代码 `F.kl_div(log_softmax(student/T), softmax(teacher/T))` 实际计算 KL(teacher || student)，config 中相反方向的简写有误，以冻结代码和PLAN的显式损失调用为准；此处仅纠正文案，不修改配置或重训。

source软目标是教师训练样本上的in-sample输出；教师checkpoint既往已使用同一valid选模。打乱目标在高置信教师下可形成强烈错误监督，优于该控制不等于比CE有额外知识增益。当前wide+BN单模型三seed均值89.150%、历史固定投票91.569%继续保留为基线和集成候选，不能把单个最优seed替代三seed目标。暂不据本次结果自动扫描蒸馏权重/温度或启动未来评分。
