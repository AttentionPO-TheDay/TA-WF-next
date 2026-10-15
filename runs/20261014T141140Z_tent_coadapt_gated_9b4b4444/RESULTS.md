# tent_coadapt_gated 结果

9个新适应单元与63条历史参考完成；独立postrun审计通过。

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

## 配对差异（coadapt − 参考）

| 参考 | Macro-F1差(pp) | Accuracy差(pp) | NAR差(pp) | 纠错率差(pp) | F1提升单元 |
|---|---:|---:|---:|---:|---:|
| tent_bn | -3.790 | -3.066 | +7.893 | +9.964 | 0/9 |
| tent_entropy_gate | -4.237 | -3.635 | +8.787 | +9.909 | 0/9 |
| tent_relative_entropy_gate | -3.244 | -2.463 | +7.341 | +10.431 | 0/9 |
| tent_fixedstats_ln | -3.987 | -3.358 | +7.749 | +8.055 | 0/9 |
| no_pseudo_fixed | -6.420 | -5.365 | +8.051 | +1.956 | 0/9 |
| frozen | -3.773 | -3.000 | +12.014 | +20.920 | 0/9 |
| bn_batchstats_no_update | -3.008 | -2.163 | +8.360 | +15.190 | 0/9 |

主判据 Macro-F1>72%：FAIL；相对原版tent_bn增益目标+1.5pp：FAIL。
窄门控平均保留样本比例：1.59%；全参参数量：1048838；source适应前向/反向暴露：0。
该组只支持联合配方的TemporalDrift开发证据，不能单独证明BN支路失步、各组件因果或跨算法普适性。no_pseudo_fixed含源域回放，作为历史参考而非完全同协议对照。

## 资源与塌缩诊断

| 条件 | 预测类别数范围 | 最大类占比范围 | 有限logits |
|---|---:|---:|---|
| tent_coadapt_gated | 101–102 | 1.724%–4.283% | True |
