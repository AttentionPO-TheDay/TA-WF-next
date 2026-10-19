# Table 3: External method transfer (reused v4)

各外部历史数据独立训练方向模型，非Temporal闭集分类器跨类别零样本迁移；新prototype_anchor未在本表评价。切片×seed等权。

| Dataset | Method | F1 (%) | Accuracy (%) | ΔF1 (pp) | NAR (%) | Positive / total |
|---|---|---:|---:|---:|---:|---:|
| wtt_time | frozen | 77.746 | 79.765 | +0.000 | 0.000 | 0/21 |
| wtt_time | tent_entropy_gate | 78.547 | 80.547 | +0.800 | 1.474 | 19/21 |
| wtt_time | token_adain | 77.970 | 79.969 | +0.223 | 0.465 | 20/21 |
| wtt_time | token_projector | 78.096 | 80.062 | +0.350 | 1.319 | 17/21 |
| awf | frozen | 82.959 | 88.430 | +0.000 | 0.000 | 0/15 |
| awf | tent_entropy_gate | 83.223 | 88.782 | +0.264 | 0.924 | 14/15 |
| awf | token_adain | 82.886 | 88.363 | -0.073 | 0.416 | 2/15 |
| awf | token_projector | 82.863 | 88.332 | -0.096 | 0.521 | 2/15 |

WTT 2022-02-16 frozen F1=18.545%，严重退化原因未证实，全部日期与负收益均保留。AWF按source定义200类评分，未知标签排除口径沿用v4。
本轮选择原型方法受到上述外部反馈影响；登记为post-external-feedback版本，不再把该测试部分视作新方法未反馈确认集。
原始144单元与完整逐日期表：`runs/20261018T130529Z_p2_external_transfer_v3_pure_fa94a001/RESULTS.md`。
