# 实验结果

状态：completed；CPU四条件×三seed，source2040/valid510，100轮上限。

| 条件 | seed | 状态 | best epoch | train accuracy | valid accuracy | valid Macro-F1 | 秒 |
|---|---:|---|---:|---:|---:|---:|---:|
| R00_baseline | 11729 | completed | 100 | 73.333% | 24.118% | 22.548% | 1395.3 |
| R00_baseline | 13407 | completed | 50 | 45.833% | 20.588% | 19.328% | 1383.2 |
| R00_baseline | 12026 | completed | 70 | 59.020% | 23.725% | 22.111% | 1411.5 |
| R01_classifier_reg | 11729 | completed | 40 | 21.275% | 18.039% | 13.280% | 1569.7 |
| R01_classifier_reg | 13407 | completed | 70 | 31.373% | 20.196% | 16.726% | 1635.8 |
| R01_classifier_reg | 12026 | completed | 100 | 37.745% | 17.059% | 15.180% | 1532.7 |
| R10_generator_anchor | 11729 | completed | 85 | 59.118% | 22.353% | 21.426% | 1359.2 |
| R10_generator_anchor | 13407 | completed | 100 | 75.833% | 25.686% | 24.294% | 1394.1 |
| R10_generator_anchor | 12026 | completed | 85 | 72.990% | 24.902% | 23.366% | 1407.8 |
| R11_both | 11729 | completed | 55 | 22.402% | 17.451% | 13.646% | 1595.3 |
| R11_both | 13407 | completed | 60 | 22.794% | 20.980% | 17.233% | 1581.7 |
| R11_both | 12026 | completed | 95 | 36.176% | 20.392% | 18.198% | 1595.0 |

| 条件 | mean valid accuracy | mean valid Macro-F1 |
|---|---:|---:|
| R00_baseline | 22.810% | 21.329% |
| R01_classifier_reg | 18.431% | 15.062% |
| R10_generator_anchor | 24.314% | 23.029% |
| R11_both | 19.608% | 16.359% |

R01_classifier_reg − R00_baseline：F1 -6.267pp，accuracy -4.379pp；逐seed F1差 -9.268, -2.603, -6.932pp；预定门槛 FAIL。

R10_generator_anchor − R00_baseline：F1 +1.700pp，accuracy +1.503pp；逐seed F1差 -1.122, +4.966, +1.255pp；预定门槛 FAIL。

R11_both − R00_baseline：F1 -4.970pp，accuracy -3.203pp；逐seed F1差 -8.903, -2.095, -3.913pp；预定门槛 FAIL。

R11_both − R01_classifier_reg：F1 +1.297pp，accuracy +1.176pp；逐seed F1差 +0.365, +0.507, +3.019pp；预定门槛 PASS。

R11_both − R10_generator_anchor：F1 -6.670pp，accuracy -4.706pp；逐seed F1差 -7.781, -7.061, -5.167pp；预定门槛 FAIL。

2×2效应（valid F1，描述性，单位pp）：
generator_main_effect_pp: [-0.378216818128438, 2.7367266182828054, 2.136727126468241]，均值 +1.498。
classifier_main_effect_pp: [-8.524317929138398, -4.8321509906148385, -6.049463831273787]，均值 -6.469。
interaction_pp: [1.4867073698599107, -4.458471799620952, 1.7642354420422066]，均值 -0.403。

| 条件 | best source accuracy | best source F1 | source−valid accuracy gap pp | best G偏移L2 | last G偏移L2 |
|---|---:|---:|---:|---:|---:|
| R00_baseline | 59.395% | 58.227% | 36.585 | 5.462 | 6.420 |
| R01_classifier_reg | 30.131% | 25.605% | 11.699 | 5.357 | 6.747 |
| R10_generator_anchor | 69.314% | 68.627% | 45.000 | 3.338 | 3.512 |
| R11_both | 27.124% | 23.640% | 7.516 | 2.260 | 2.676 |

解释：分类器正则化是dropout和weight_decay的组合干预；锚定只限制生成器相对其随机初始化的偏移。训练拟合或差距下降本身不是成功，须看valid是否稳定上升；相关主效应不证明唯一过拟合来源。三seed共享已观察valid，结果仅为固定强度的开发筛查，未搜索超参或访问外部测试。

独立核验：24/24；errors=[]。

参数量、可训练参数、生成器反馈梯度/token变化见各seed metrics/history。代码与输入散列见artifacts/freeze.json、input_audit.json；检查、合成吞吐、训练及验证分开留存。每epoch保存完整恢复状态。

仅使用TemporalDrift已观察source/valid；所有新模型从头初始化。无GPU、未来日期或外部数据访问。未追加机制或超参数搜索。

## 结果解释补充

本轮没有任何新条件稳定超过同轮R00基线。唯一PASS是R11相对已明显受损的R01，不能据此采用组合方案：R11仍比R00低4.970pp F1。

分类器强正则化使训练拟合和valid性能同时下降，提示该强度组合在本预算下限制了学习；不是泛化改善，也不能区分dropout与weight_decay的单项责任。生成器锚定使末轮参数偏移L2从6.420降至3.512，说明约束实际生效；单独启用的valid F1平均+1.700pp，但一个seed为负，尚未达到稳定收益门槛。该方向可以保留为候选，不能宣称过拟合来源已定位。

表中source指标对应各自按valid选择的best checkpoint，而非统一末轮。本轮baseline的末轮source accuracy为75.752%，锚定组74.510%；best checkpoint source accuracy分别为59.395%与69.314%，不能把不同选模epoch的差异直接解释为约束增强记忆。使用的是新的模型训练seeds，历史77.55%的固定生成器重训结果亦属不同训练流程。

当前建议：不采用本轮强分类器正则化或组合方案；生成器锚定仅保留待验证。未自动追加条件、调参或新训练。
