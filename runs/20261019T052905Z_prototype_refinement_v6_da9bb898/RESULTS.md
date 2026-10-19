# 原型细化 v6 结果

36次新适应和全量重载审计完成，得到小幅提分，但尚未改善全部风险区间。主候选sharp_target_only相对v5动量原型：Version045 ΔF1从+2.718→+2.903pp（额外+.185pp），USA从+1.553→+1.680pp（额外+.127pp）；未达到预定两片各+.20pp标准，也未达到两片各+3pp的扩展门槛。

| 切片 | v5→v6 NAR % | v5→v6 Recovery % | 同NAR≤1%纠错 % | 同NAR≤2%纠错 % |
|---|---:|---:|---:|---:|
| Version045 | 1.975→2.136 | 8.445→9.185 | 4.367→4.402 | 7.991→8.512 |
| USA | 2.338→2.337 | 6.544→6.902 | 3.087→2.810 | 5.521→6.031 |

USA在近似相同raw NAR下纠错+.358pp，存在正向诊断信号；但NAR≤1%的低风险端纠错−.277pp，不能称为前沿全面支配。NAR≤2%端两片事后纠错均增加约.51–.52pp，仍只是有标签定位的诊断曲线，不是无标签部署保证。主候选相对v5在6个seed×slice中5个F1增加，Version seed2026为−.019pp；USA整体ΔF1 SD仍达.893pp，未消除种子差异。以上是描述性开发结果，不宣称统计显著。

36/36新适应完成；预定主候选sharp_target_only配对改进门槛：False；扩展资格：False。

[全部方法与逐切片主表](table_main.md)；[风险—纠错前沿](figure_frontier.png)；[原型移动轨迹](figure_prototype.png)。

对照为已审计v5动量原型的历史复用。所有新组采用相同纯净底座、分区、400步及源identity预算。没有新增源模型训练。

version_045：
- anchor_only：ΔF1 +2.130pp，较v5 -0.588pp；NAR变化+0.120pp；同NAR1%纠错变化-0.577pp。
- target_only：ΔF1 +2.789pp，较v5 +0.071pp；NAR变化-0.008pp；同NAR1%纠错变化+0.081pp。
- fixed_geometry：ΔF1 +2.153pp，较v5 -0.565pp；NAR变化+0.385pp；同NAR1%纠错变化-1.177pp。
- sharp_proto：ΔF1 +2.876pp，较v5 +0.158pp；NAR变化+0.179pp；同NAR1%纠错变化-0.004pp。
- radius_proto：ΔF1 +2.708pp，较v5 -0.010pp；NAR变化+0.018pp；同NAR1%纠错变化-0.054pp。
- sharp_target_only：ΔF1 +2.903pp，较v5 +0.185pp；NAR变化+0.161pp；同NAR1%纠错变化+0.034pp。

network_USA：
- anchor_only：ΔF1 +1.381pp，较v5 -0.172pp；NAR变化-0.270pp；同NAR1%纠错变化+0.326pp。
- target_only：ΔF1 +1.514pp，较v5 -0.038pp；NAR变化+0.046pp；同NAR1%纠错变化-0.073pp。
- fixed_geometry：ΔF1 +1.242pp，较v5 -0.311pp；NAR变化+0.294pp；同NAR1%纠错变化-1.194pp。
- sharp_proto：ΔF1 +1.677pp，较v5 +0.125pp；NAR变化-0.001pp；同NAR1%纠错变化-0.455pp。
- radius_proto：ΔF1 +1.551pp，较v5 -0.002pp；NAR变化+0.000pp；同NAR1%纠错变化+0.000pp。
- sharp_target_only：ΔF1 +1.680pp，较v5 +0.127pp；NAR变化-0.001pp；同NAR1%纠错变化-0.277pp。

所有臂、三个源训练seed和两切片完整保留。适应初始化/批流均固定20261019，源训练seed不等于独立适应随机流；均值SD仅描述稳定性。

曲线用query标签事后定位阈值，只是诊断前沿，不能把对应点当作可部署策略或风险保证。实际报告raw预测无冻结回退，source-valid仅终端评分；query不参与原型/记忆拟合。

本轮受v5公开切片结果反馈影响，仅为两切片诊断开发证据；Version身份与跨域类别采用已有内容支持假设，Network仅地理切片，不归因丢包或定量时延。未进行WTT/AWF新评分，未启动全矩阵。

预算36新任务固定结束；任何最佳探索性组不能替换预定主候选。若未改善或未达+3pp目标，保留阴性，不继续追加搜索。

36个checkpoint的完整query和source-valid重载、逐指标独立复算、纯净底座锁定、reference状态范围及输入/输出SHA核验通过。

机制归因与事后误差诊断（实验完成后追加；不修改任何训练配置、预测或阈值）：

- **动态几何目标优于只移动loss锚点**：target_only相对anchor_only平均F1增加+.659pp（Version）/+.133pp（USA）；但两组同时改变目标和loss锚点，不能把该差值当作单因素因果效应。更直接的匹配比较是v5 proto vs anchor_only（固定动态loss锚点开关，仅切换几何目标，同时原型状态会随目标演化）：+.588/+.171pp；target_only vs fixed_geometry则为+.636/+.272pp，支持reference动态几何信号优于固定源几何配方。
- **固定student源锚点并非普适增益**：target_only vs v5 proto：Version+.071pp，USA−.039pp；锐化后sharp_target_only vs sharp_proto仅+.027/+.003pp，说明本轮均值提分主要关联几何锐化，而非分离锚点本身。
- **保持源类半径没有帮助**：radius_proto相对v5约−.010/−.002pp；观察到原型收缩不足以证明它是主要瓶颈。这只否定本轮半径恢复配方，不排除更复杂拓扑变化。
- **独立纠错证据稀缺**：源分类器原本错误的样本中，约65.4%（Version）/65.1%（USA）与最近源原型指向同一错误类别；最近原型正确仅约10.35%/10.17%。约31.9%/30.5%的原错误样本softmax置信度≥.9，而v6仅能修复其中.729%/.333%。这些是按三个源seed平均的误差条件统计，不能外推为全部样本占比或信息不可恢复证明。

后续研究优先级：保留动态几何与锐化作为候选，先解决“分类器与源几何共同错”的纠错证据来源；不再把恢复原型半径或追加同类门控作为已证实的突破口。低风险端路由仍需单独解决，不能为提高均值牺牲NAR≤1%端。当前不启动全量扩展。

[事后误差分组](artifacts/posthoc_error_group_summary.json)及[逐seed配对变化](artifacts/paired_primary.json)来自已审计保存预测，posthoc_diagnostics.py可重现。此段只追加结果解释；冻结的主候选、方法、预算和裁决保持原样。交付SHA因此重新核验。
