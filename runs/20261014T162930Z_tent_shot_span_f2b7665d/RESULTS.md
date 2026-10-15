# tent_shot_span 结果

9个新适应单元与72条历史参考完成；独立postrun审计通过。

| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |
|---|---:|---:|---:|---:|
| frozen | 70.857% | 72.579% | 0.000% | 0.000% |
| no_pseudo_fixed | 73.504% | 74.943% | 3.963% | 18.964% |
| tent_bn | 70.874% | 72.645% | 4.122% | 10.956% |
| bn_batchstats_no_update | 70.091% | 71.742% | 3.655% | 5.730% |
| tent_fixedstats_ln | 71.070% | 72.936% | 4.266% | 12.865% |
| tent_entropy_gate | 71.320% | 73.213% | 3.228% | 11.011% |
| tent_relative_entropy_gate | 70.328% | 72.042% | 4.673% | 10.489% |
| tent_coadapt_gated | 67.083% | 69.578% | 12.014% | 20.920% |
| tent_shot_span | 70.875% | 72.856% | 4.637% | 13.724% |

## day14

| 条件 | Macro-F1 | 负适应率 | 纠错率 |
|---|---:|---:|---:|
| frozen | 86.289% | 0.000% | 0.000% |
| no_pseudo_fixed | 86.642% | 2.757% | 19.893% |
| tent_bn | 86.076% | 2.239% | 12.938% |
| bn_batchstats_no_update | 86.143% | 1.107% | 5.968% |
| tent_fixedstats_ln | 85.709% | 2.968% | 15.332% |
| tent_entropy_gate | 86.186% | 2.162% | 13.235% |
| tent_relative_entropy_gate | 85.296% | 3.078% | 12.794% |
| tent_coadapt_gated | 82.488% | 8.148% | 25.341% |
| tent_shot_span | 85.543% | 3.369% | 16.856% |

## day90

| 条件 | Macro-F1 | 负适应率 | 纠错率 |
|---|---:|---:|---:|
| frozen | 69.459% | 0.000% | 0.000% |
| no_pseudo_fixed | 72.346% | 4.109% | 18.440% |
| tent_bn | 69.396% | 3.936% | 9.538% |
| bn_batchstats_no_update | 68.941% | 2.927% | 5.390% |
| tent_fixedstats_ln | 69.722% | 4.501% | 11.978% |
| tent_entropy_gate | 69.771% | 3.213% | 9.446% |
| tent_relative_entropy_gate | 69.427% | 4.279% | 9.874% |
| tent_coadapt_gated | 65.425% | 12.929% | 19.235% |
| tent_shot_span | 69.209% | 5.092% | 12.180% |

## day270

| 条件 | Macro-F1 | 负适应率 | 纠错率 |
|---|---:|---:|---:|
| frozen | 56.823% | 0.000% | 0.000% |
| no_pseudo_fixed | 61.524% | 5.025% | 18.557% |
| tent_bn | 57.150% | 6.190% | 10.393% |
| bn_batchstats_no_update | 55.190% | 6.930% | 5.832% |
| tent_fixedstats_ln | 57.780% | 5.328% | 11.285% |
| tent_entropy_gate | 58.003% | 4.308% | 10.353% |
| tent_relative_entropy_gate | 56.260% | 6.663% | 8.799% |
| tent_coadapt_gated | 53.337% | 14.966% | 18.183% |
| tent_shot_span | 57.873% | 5.448% | 12.134% |

## 配对差异（shot_span − 参考）

| 参考 | Macro-F1差(pp) | Accuracy差(pp) | NAR差(pp) | 纠错率差(pp) | F1提升单元 |
|---|---:|---:|---:|---:|---:|
| tent_bn | +0.001 | +0.211 | +0.515 | +2.767 | 4/9 |
| tent_entropy_gate | -0.445 | -0.358 | +1.409 | +2.713 | 1/9 |
| tent_relative_entropy_gate | +0.547 | +0.814 | -0.037 | +3.234 | 8/9 |
| tent_fixedstats_ln | -0.195 | -0.080 | +0.371 | +0.859 | 2/9 |
| tent_coadapt_gated | +3.792 | +3.277 | -7.378 | -7.196 | 9/9 |
| no_pseudo_fixed | -2.629 | -2.087 | +0.673 | -5.240 | 0/9 |
| frozen | +0.018 | +0.277 | +4.637 | +13.724 | 4/9 |
| bn_batchstats_no_update | +0.784 | +1.114 | +0.982 | +7.994 | 4/9 |

主判据 Macro-F1>72%：FAIL；相对原版tent_bn增益目标+1.5pp：FAIL。
平均保留样本比例：8.19%；双视图预测一致率：91.86%；可训练参数量：32384；BN统计更新：0。
该组是Scale-Lock与Span-Invariance联合配方的TemporalDrift开发证据，不能单独证明两组件的因果贡献或跨数据集普适性。

## 资源与塌缩诊断

| 条件 | 预测类别数范围 | 最大类占比范围 | 有限logits |
|---|---:|---:|---|
| tent_shot_span | 102–102 | 1.381%–2.966% | True |
