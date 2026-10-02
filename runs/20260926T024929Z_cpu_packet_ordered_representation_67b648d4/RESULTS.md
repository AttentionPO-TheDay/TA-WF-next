# 实验结果

状态：completed；CPU两条件×三seed，source2040/valid510，方向5000包，100epochs。

| 条件 | seed | 完成状态 | best epoch | train accuracy | valid accuracy | valid Macro-F1 | 秒 |
|---|---:|---|---:|---:|---:|---:|---:|
| summary | 1729 | completed | 100 | 68.039% | 25.686% | 23.773% | 977.7 |
| summary | 3407 | completed | 75 | 48.676% | 24.902% | 23.143% | 993.1 |
| summary | 2026 | completed | 100 | 62.647% | 28.824% | 27.048% | 979.3 |
| ordered | 1729 | completed | 75 | 71.765% | 18.627% | 17.359% | 983.2 |
| ordered | 3407 | completed | 60 | 47.010% | 17.843% | 16.444% | 994.7 |
| ordered | 2026 | completed | 70 | 66.127% | 19.020% | 17.683% | 990.9 |

summary/ordered 三seed均值：accuracy 26.471/18.497%；Macro-F1 24.655/17.162%。
ordered-summary F1差 -7.493pp；逐seed差 -6.414, -6.699, -9.365pp；accuracy差 -7.974pp。
预定门槛（平均F1至少+1pp、3/3正、accuracy不下降）：FAIL_CANDIDATE。

结论：该受控表示改动未通过预定候选门槛，不支持直接推进后续预训练或TTA。报告保留全部正负seed；不能由一轮否定所有保序编码或Transformer。下一步优先根据训练拟合情况区分优化/汇聚瓶颈与泛化，另立有界方案。

核验：12/12 source/valid预测与checkpoint、独立指标、最早最佳epoch复算；errors=[]。

输入/代码版本见 config.json、artifacts/freeze.json、artifacts/input_audit.json。两条件均118870参数、同seed初始化一致、run/window不变；增加的是packet输入细节，摘要对照具有更低有效输入秩。每轮保存latest及按valid改进保存best。

历史DF accuracy49.150%/F1 48.144%只是不同架构及训练/选模预算的性能参照，本轮未重训DF/RF。仅TemporalDrift已观察开发数据；未来日期、WTT/AWF、PCAP、预训练、蒸馏、adapter/TTA均未使用。
