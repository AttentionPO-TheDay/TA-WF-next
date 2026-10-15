# TENT 实验结果

18个新单元与18条历史参考完成，postrun审计通过。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| no_pseudo_fixed | 73.504% | 74.943% | 3.963% | 18.964% |
| tent_bn | 70.874% | 72.645% | 4.122% | 10.956% |
| bn_batchstats_no_update | 70.091% | 71.742% | 3.655% | 5.730% |

## day14

| 条件 | Macro-F1 | 负适应率 | 纠错率 |
|---|---:|---:|---:|
| frozen | 86.289% | 0.000% | 0.000% |
| no_pseudo_fixed | 86.642% | 2.757% | 19.893% |
| tent_bn | 86.076% | 2.239% | 12.938% |
| bn_batchstats_no_update | 86.143% | 1.107% | 5.968% |

## day90

| 条件 | Macro-F1 | 负适应率 | 纠错率 |
|---|---:|---:|---:|
| frozen | 69.459% | 0.000% | 0.000% |
| no_pseudo_fixed | 72.346% | 4.109% | 18.440% |
| tent_bn | 69.396% | 3.936% | 9.538% |
| bn_batchstats_no_update | 68.941% | 2.927% | 5.390% |

## day270

| 条件 | Macro-F1 | 负适应率 | 纠错率 |
|---|---:|---:|---:|
| frozen | 56.823% | 0.000% | 0.000% |
| no_pseudo_fixed | 61.524% | 5.025% | 18.557% |
| tent_bn | 57.150% | 6.190% | 10.393% |
| bn_batchstats_no_update | 55.190% | 6.930% | 5.832% |

tent_bn：平均任务秒 41.496；query摊销毫秒/trace 0.155；峰值显存GiB 0.987；可训练参数 1792。

bn_batchstats_no_update：平均任务秒 22.965；query摊销毫秒/trace 0.149；峰值显存GiB 0.987；可训练参数 0。

TENT 使用纯 softmax entropy 最小化；tent_bn 更新 BN1d affine 参数，LN 和其他权重冻结。
bn_batchstats_no_update 使用相同 BN batch-statistics 前向但不执行熵梯度更新，用于分离批统计变化与参数更新。
该结果是 TemporalDrift 已观察开发证据；不据此声称所有主流TTA均失败。
{
  "tent_bn": {
    "macro_f1": 0.7087399099627114,
    "negative_adaptation_rate": 0.04121730861408554,
    "recovery_rate": 0.10956163664482738,
    "f1_delta_vs_no_pseudo_fixed_pp": -2.629835594586416,
    "nar_delta_vs_no_pseudo_fixed_pp": 0.15826722219891842,
    "recovery_delta_vs_no_pseudo_fixed_pp": -8.007393382961736,
    "positive_paired_cells": 0,
    "screen_pass": false
  },
  "bn_batchstats_no_update": {
    "macro_f1": 0.7009132799017845,
    "negative_adaptation_rate": 0.03654934322459212,
    "recovery_rate": 0.05729861222248419,
    "f1_delta_vs_no_pseudo_fixed_pp": -3.4124986006791036,
    "nar_delta_vs_no_pseudo_fixed_pp": -0.30852931675042355,
    "recovery_delta_vs_no_pseudo_fixed_pp": -13.233695825196055,
    "positive_paired_cells": 0,
    "screen_pass": false
  }
}

## 配对结论与适用范围

补充说明：原 runner 完成审计但遗漏 report() 调用，RESULTS.md 原先仍为模板。本次仅从已审计的 scores.json 生成摘要，不重新适应或修改冻结配置。

| TENT 相对条件 | Macro-F1差(pp) | NAR差(pp) | 纠错率差(pp) | F1提升单元 |
|---|---:|---:|---:|---:|
| bn_batchstats_no_update | +0.783 | +0.467 | +5.226 | 8/9 |
| frozen | +0.017 | +4.122 | +10.956 | 5/9 |
| no_pseudo_fixed | -2.630 | +0.158 | -8.007 | 0/9 |

主归因比较为 tent_bn 对 bn_batchstats_no_update；frozen/no_pseudo_fixed 是历史描述参考，BN推理统计规则不同。query不参与梯度更新，但BN使用query批次统计；本实验是官方TENT核心机制的BN1d移植及support→query评价，不是官方在线同query更新的端到端复现。

负适应率以源模型原本正确的样本为分母，纠错率以源模型原本错误的样本为分母；表中数值均为3 seed×3日期的9单元等权均值。不能将report()中的screen_pass作为预注册主判据；它沿用了上次Soft Pacing的3%/纠错维持筛查，本次主结论以配对差值为准。

重载审计逐checkpoint复核首个128条query logits（共18个），而指标重算覆盖全部36条结果；不声称重载预测覆盖整个query。GPU1为共享GPU，耗时仅描述。

| 条件 | 预测类别数范围 | 最大类别占比范围 | 有限logits |
|---|---:|---:|---|
| tent_bn | 102–102 | 1.237%–2.279% | True |
| bn_batchstats_no_update | 102–102 | 1.172%–2.284% | True |

未访问WTT-Time/AWF，结果仅是TemporalDrift开发证据。失败版本及实现修复均保留，未以性能反馈改变超参数。
