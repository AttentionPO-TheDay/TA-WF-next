# 外部独立方向模型方法迁移 v4

144 个单元完成；全部预测文件、checkpoint/对齐层重载及 sklearn 指标复算通过。此结果检验冻结方法迁移，不是 Temporal 唯一模型的零样本跨类别分类。AWF 是固定文件诊断评价；WTT 不在未经完整暴露审计时宣称严格未见确认。

| Dataset | Slice | Condition | F1 mean±sd (%) | Accuracy (%) | ΔF1 (pp) | NAR (%) |
|---|---|---|---:|---:|---:|---:|
| awf | tor_time_test10d_200w_100tr.npz | frozen | 89.425±0.264 | 94.411 | +0.000 | 0.000 |
| awf | tor_time_test10d_200w_100tr.npz | tent_entropy_gate | 89.648±0.244 | 94.658 | +0.223 | 0.634 |
| awf | tor_time_test10d_200w_100tr.npz | token_adain | 89.437±0.268 | 94.404 | +0.012 | 0.201 |
| awf | tor_time_test10d_200w_100tr.npz | token_projector | 89.396±0.250 | 94.369 | -0.029 | 0.298 |
| awf | tor_time_test2w_200w_100tr.npz | frozen | 82.450±0.701 | 88.029 | +0.000 | 0.000 |
| awf | tor_time_test2w_200w_100tr.npz | tent_entropy_gate | 82.799±0.816 | 88.497 | +0.348 | 0.804 |
| awf | tor_time_test2w_200w_100tr.npz | token_adain | 82.366±0.640 | 87.932 | -0.085 | 0.328 |
| awf | tor_time_test2w_200w_100tr.npz | token_projector | 82.338±0.729 | 87.880 | -0.112 | 0.418 |
| awf | tor_time_test3d_200w_100tr.npz | frozen | 92.081±0.447 | 96.893 | +0.000 | 0.000 |
| awf | tor_time_test3d_200w_100tr.npz | tent_entropy_gate | 92.281±0.368 | 97.126 | +0.200 | 0.374 |
| awf | tor_time_test3d_200w_100tr.npz | token_adain | 92.030±0.484 | 96.838 | -0.052 | 0.145 |
| awf | tor_time_test3d_200w_100tr.npz | token_projector | 92.022±0.475 | 96.828 | -0.060 | 0.181 |
| awf | tor_time_test4w_200w_100tr.npz | frozen | 77.317±0.843 | 82.827 | +0.000 | 0.000 |
| awf | tor_time_test4w_200w_100tr.npz | tent_entropy_gate | 77.440±1.055 | 83.039 | +0.123 | 1.484 |
| awf | tor_time_test4w_200w_100tr.npz | token_adain | 77.263±0.919 | 82.784 | -0.053 | 0.595 |
| awf | tor_time_test4w_200w_100tr.npz | token_projector | 77.205±0.897 | 82.746 | -0.112 | 0.768 |
| awf | tor_time_test6w_200w_100tr.npz | frozen | 73.523±0.929 | 79.990 | +0.000 | 0.000 |
| awf | tor_time_test6w_200w_100tr.npz | tent_entropy_gate | 73.949±0.796 | 80.588 | +0.426 | 1.323 |
| awf | tor_time_test6w_200w_100tr.npz | token_adain | 73.334±0.861 | 79.855 | -0.189 | 0.809 |
| awf | tor_time_test6w_200w_100tr.npz | token_projector | 73.356±0.923 | 79.838 | -0.167 | 0.938 |
| wtt_time | 2022-01-16 | frozen | 93.032±1.562 | 94.139 | +0.000 | 0.000 |
| wtt_time | 2022-01-16 | tent_entropy_gate | 93.952±1.108 | 94.971 | +0.920 | 0.304 |
| wtt_time | 2022-01-16 | token_adain | 93.362±1.698 | 94.410 | +0.330 | 0.145 |
| wtt_time | 2022-01-16 | token_projector | 93.181±1.552 | 94.241 | +0.149 | 0.450 |
| wtt_time | 2022-01-23 | frozen | 90.022±0.868 | 91.679 | +0.000 | 0.000 |
| wtt_time | 2022-01-23 | tent_entropy_gate | 90.555±0.571 | 92.170 | +0.532 | 0.610 |
| wtt_time | 2022-01-23 | token_adain | 90.198±0.704 | 91.832 | +0.175 | 0.240 |
| wtt_time | 2022-01-23 | token_projector | 90.381±0.300 | 91.984 | +0.359 | 0.461 |
| wtt_time | 2022-01-30 | frozen | 87.985±0.843 | 90.196 | +0.000 | 0.000 |
| wtt_time | 2022-01-30 | tent_entropy_gate | 88.506±0.277 | 90.686 | +0.521 | 0.677 |
| wtt_time | 2022-01-30 | token_adain | 88.136±0.650 | 90.331 | +0.151 | 0.225 |
| wtt_time | 2022-01-30 | token_projector | 88.420±0.410 | 90.585 | +0.435 | 0.506 |
| wtt_time | 2022-02-06 | frozen | 86.249±0.821 | 88.838 | +0.000 | 0.000 |
| wtt_time | 2022-02-06 | tent_entropy_gate | 87.006±0.510 | 89.517 | +0.758 | 0.650 |
| wtt_time | 2022-02-06 | token_adain | 86.482±0.678 | 89.042 | +0.233 | 0.153 |
| wtt_time | 2022-02-06 | token_projector | 86.583±0.431 | 89.126 | +0.334 | 0.286 |
| wtt_time | 2022-02-16 | frozen | 18.545±0.715 | 21.108 | +0.000 | 0.000 |
| wtt_time | 2022-02-16 | tent_entropy_gate | 18.888±0.571 | 21.316 | +0.343 | 6.048 |
| wtt_time | 2022-02-16 | token_adain | 18.748±0.650 | 21.212 | +0.204 | 1.648 |
| wtt_time | 2022-02-16 | token_projector | 18.745±0.767 | 20.935 | +0.200 | 5.399 |
| wtt_time | 2022-02-21 | frozen | 88.095±1.483 | 89.729 | +0.000 | 0.000 |
| wtt_time | 2022-02-21 | tent_entropy_gate | 88.616±0.794 | 90.338 | +0.520 | 0.795 |
| wtt_time | 2022-02-21 | token_adain | 88.234±1.439 | 89.882 | +0.139 | 0.227 |
| wtt_time | 2022-02-21 | token_projector | 88.188±1.178 | 89.882 | +0.093 | 0.794 |
| wtt_time | 2022-04-03 | frozen | 80.297±2.093 | 82.666 | +0.000 | 0.000 |
| wtt_time | 2022-04-03 | tent_entropy_gate | 82.306±1.168 | 84.829 | +2.009 | 1.235 |
| wtt_time | 2022-04-03 | token_adain | 80.629±2.071 | 83.071 | +0.333 | 0.615 |
| wtt_time | 2022-04-03 | token_projector | 81.176±1.904 | 83.680 | +0.879 | 1.339 |

## 数据集汇总与结果解释

复核后新增汇总：切片与seed等权；NAR为单元内比率的等权均值，不是拼接样本后重算。已有逐切片数值和判定未修改。

| 数据集 | 条件 | Macro-F1 (%) | Accuracy (%) | ΔF1 (pp) | NAR (%) | ΔF1为正单元 |
|---|---|---:|---:|---:|---:|---:|
| wtt_time | frozen | 77.746 | 79.765 | +0.000 | 0.000 | 0/21 |
| wtt_time | tent_entropy_gate | 78.547 | 80.547 | +0.800 | 1.474 | 19/21 |
| wtt_time | token_adain | 77.970 | 79.969 | +0.223 | 0.465 | 20/21 |
| wtt_time | token_projector | 78.096 | 80.062 | +0.350 | 1.319 | 17/21 |
| awf | frozen | 82.959 | 88.430 | +0.000 | 0.000 | 0/15 |
| awf | tent_entropy_gate | 83.223 | 88.782 | +0.264 | 0.924 | 14/15 |
| awf | token_adain | 82.886 | 88.363 | -0.073 | 0.416 | 2/15 |
| awf | token_projector | 82.863 | 88.332 | -0.096 | 0.521 | 2/15 |

结论：当前熵门控配方在WTT与AWF分别平均挽回0.800和0.264pp Macro-F1；Token AdaIN与残差投影在WTT有小幅收益，但在AWF均值为负，且三个seed的平均ΔF1均为负。当前证据不支持Token对齐普遍优于门控，亦不支持“绝不增加误杀”；所有非frozen方法都存在原正确样本被改错。仅检验该次独立方向模型上的冻结配方，不否定所有Token对齐方法。

异常切片：WTT 2022-02-16 frozen Macro-F1为18.545%，gate为18.888%，AdaIN为18.748%，projector为18.745%；当前方法均未修复严重退化。冻结角色/类别/有效输入检查和预测重载通过不能独立证明退化的物理原因；需另立只读诊断，不能先把它归因于丢包/浏览器升级。

边界：AWF类表由源文件固定为200类，各目标文件有1000条未知标签行已排除并记录；本表Macro-F1仍按固定200类集计算，不能直接与190共同类别口径对比。外部模型在各自历史训练数据上独立从头训练，因此本实验属于方法迁移评价。实验一/三非时间因素目标评分与Temporal纯净底座的目标A/B/C尚未完成，不能把144单元当成四模块全部完成。

当前外部future反馈已被读取与汇总；据此调整对齐结构、阈值或步数须登记新方法版本和数据暴露，本测试部分不能继续充当新版本未反馈的确认集。
