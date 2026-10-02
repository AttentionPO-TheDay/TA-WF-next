# 实验结果

状态：completed；CPU四条件×三seed，source2040/valid510，100轮上限。

| 条件 | seed | 状态 | best epoch | train accuracy | valid accuracy | valid Macro-F1 | 秒 |
|---|---:|---|---:|---:|---:|---:|---:|
| A_local_mean_joint | 1729 | completed | 80 | 76.618% | 22.745% | 21.397% | 1400.5 |
| A_local_mean_joint | 3407 | completed | 55 | 43.431% | 23.137% | 21.450% | 1425.4 |
| A_local_mean_joint | 2026 | completed | 60 | 60.245% | 28.431% | 26.466% | 1268.9 |
| B_local_attention_joint | 1729 | completed | 65 | 65.931% | 22.941% | 21.363% | 1354.2 |
| B_local_attention_joint | 3407 | completed | 50 | 43.039% | 22.745% | 20.782% | 1290.7 |
| B_local_attention_joint | 2026 | completed | 90 | 78.137% | 28.824% | 27.540% | 1359.6 |
| C_local_attention_frozen | 1729 | completed | 90 | 57.059% | 25.294% | 23.342% | 1164.2 |
| C_local_attention_frozen | 3407 | completed | 100 | 59.510% | 26.078% | 23.511% | 1115.2 |
| C_local_attention_frozen | 2026 | completed | 80 | 57.304% | 29.608% | 28.085% | 1159.8 |
| D_local_attention_slow | 1729 | completed | 95 | 66.127% | 25.294% | 23.389% | 1365.4 |
| D_local_attention_slow | 3407 | completed | 95 | 65.392% | 24.510% | 23.956% | 1345.4 |
| D_local_attention_slow | 2026 | completed | 100 | 79.902% | 28.039% | 26.216% | 1398.5 |

| 条件 | mean valid accuracy | mean valid Macro-F1 |
|---|---:|---:|
| A_local_mean_joint | 24.771% | 23.104% |
| B_local_attention_joint | 24.837% | 23.229% |
| C_local_attention_frozen | 26.993% | 24.979% |
| D_local_attention_slow | 25.948% | 24.520% |

B_local_attention_joint − C_local_attention_frozen：F1 -1.751pp，accuracy -2.157pp；逐seed F1差 -1.979, -2.728, -0.545pp；预定门槛 FAIL。

B_local_attention_joint − A_local_mean_joint：F1 +0.124pp，accuracy +0.065pp；逐seed F1差 -0.034, -0.667, +1.074pp；预定门槛 FAIL。

D_local_attention_slow − B_local_attention_joint：F1 +1.292pp，accuracy +1.111pp；逐seed F1差 +2.026, +3.173, -1.324pp；预定门槛 FAIL。

历史摘要参照为accuracy26.471% / F1 24.655%，明确复用，不是新的独立重复。
A_local_mean_joint 相对历史摘要：F1 -1.551pp，accuracy -1.699pp；三seed F1差 [-2.3761458463698535, -1.6936677317300113, -0.5824904925706198]。
B_local_attention_joint 相对历史摘要：F1 -1.426pp，accuracy -1.634pp；三seed F1差 [-2.4101613613997515, -2.360895670629784, 0.4919775854031916]。
C_local_attention_frozen 相对历史摘要：F1 +0.324pp，accuracy +0.523pp；三seed F1差 [-0.4312268988557494, 0.36734939416600554, 1.0364957034507238]。
D_local_attention_slow 相对历史摘要：F1 -0.135pp，accuracy -0.523pp；三seed F1差 [-0.3841073837431569, 0.8124648003540791, -0.8322945476924692]。

解释：B−C仅检验分类反馈相对随机冻结生成器的作用，胜过C不能证明实际收益；B−A检验可学习汇聚；D−B检验较小生成器更新尺度。结果仅为预定开发筛查，不是统计显著性或独立确认。采用候选仍需结合历史摘要的实际性能及全部seed一致性，不能只挑有利比较。

独立核验：24/24；errors=[]。

参数量、可训练参数、生成器反馈梯度/token变化见各seed metrics/history。代码与输入散列见artifacts/freeze.json、input_audit.json；检查、合成吞吐、训练及验证分开留存。每epoch保存完整恢复状态。

仅使用TemporalDrift已观察source/valid；所有新模型从头初始化。无GPU、未来日期或外部数据访问。未追加机制或超参数搜索。
