# Day14 / Day90 冻结基线评分结果

完成；独立指标与checkpoint重载审计通过。无训练、TTA、BN更新或目标选模。

| 数据 | Accuracy均值±样本SD | Macro-F1均值±样本SD | Accuracy下降pp | F1下降pp |
|---|---:|---:|---:|---:|
| valid | 89.150 ± 0.226% | 88.944 ± 0.257% | 0.000 | 0.000 |
| day14 | 86.350 ± 0.270% | 86.294 ± 0.315% | 2.800 | 2.649 |
| day90 | 71.437 ± 0.636% | 69.425 ± 0.750% | 17.713 | 19.519 |

逐seed及每日期样本数见 artifacts/scores.json；完整预测及logits、原始行号均在artifacts。数据hash、类别计数、训练/valid/跨日期精确指纹重叠见data_manifest.json；重叠未按成绩过滤。输入和权重来源见SOURCES.md及provenance.json。

该差值以反复选模的源期510行valid作参照，不是独立Day0测试，也不是时间因素的因果效应。三seed方差不代表跨日期或数据抽样的不确定性。Day14/Day90现为本方法开发暴露；结果不能作为未经反馈的独立确认。未评价其他日期、WTT-Time或AWF。
