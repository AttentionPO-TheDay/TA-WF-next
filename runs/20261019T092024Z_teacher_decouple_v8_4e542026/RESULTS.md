# Teacher解耦与reference风险路由 v8

18/18新适应完成。预注册去KL主判据=False；固定v6一致性包络评分判据=False。

[完整主表](table_main.md)；[全部120部署评分状态](table_scoring.md)；[同风险诊断曲线](figure_frontier.png)；[几何梯度倍率轨迹](figure_gradient_weights.png)。

version_045 / no_teacher_kl：相对v6 F1 -1.196pp，NAR -1.196pp，Rec@NAR1 -1.494pp；聚类纯度 -0.270pp，共错集合纠错 -0.542pp。判据=False。

version_045 / confident_teacher_kl：相对v6 F1 -0.526pp，NAR -0.777pp，Rec@NAR1 +0.126pp；聚类纯度 +0.714pp，共错集合纠错 -0.198pp。判据=False。

version_045 / geometry_grad_balance：相对v6 F1 +0.370pp，NAR +0.028pp，Rec@NAR1 +0.808pp；聚类纯度 +0.318pp，共错集合纠错 +1.059pp。判据=True。

network_USA / no_teacher_kl：相对v6 F1 -1.399pp，NAR -1.549pp，Rec@NAR1 -2.883pp；聚类纯度 -0.671pp，共错集合纠错 -1.242pp。判据=False。

network_USA / confident_teacher_kl：相对v6 F1 -0.307pp，NAR -0.915pp，Rec@NAR1 -0.720pp；聚类纯度 +0.611pp，共错集合纠错 -0.268pp。判据=False。

network_USA / geometry_grad_balance：相对v6 F1 +0.213pp，NAR -0.137pp，Rec@NAR1 +0.084pp；聚类纯度 +0.971pp，共错集合纠错 +0.095pp。判据=False。

version_045 / 固定v6 agreement_q95：三个source-seed NAR=[0.022983222247759134, 0.13866420152530623, 0.007898894154818325]%，纠错较rank-source1变化-4.440pp。判据=False。

network_USA / 固定v6 agreement_q95：三个source-seed NAR=[0.0, 0.0, 0.06561679790026247]%，纠错较rank-source1变化-5.241pp。判据=False。

解释边界：KL去除只移除了直接输出分布约束，源邻居混合q、pull/relation几何锚点仍继承原编码器证据，不能称完全teacher-free。几何倍增不是创造独立证据；梯度范数不包含方向信息，需同时查看cosine与触发裁剪情况。

所有BN affine/运行统计和纯source底座保持原值，未运行TENT拼接、BN协同更新或梯度正交投影。固定v6控制历史适应不算本轮新重复；所有源/query/reference/validation控制预测已重新核对。

聚类只拟合reference标准化embedding102类KMeans；query majority purity/ARI/NMI只作终端诊断，不能用query类标签给聚类命名或部署。共错集合固定为原head/source-prototype/双source-bank同错，报告still-same-wrong/other-wrong/corrected，变化不等于全部可救错误。

ECDF与分位拟合只用独立reference；sourcevalid阈值规则不从query返调。无GMM及sigmoid(gain)。q99/一致性包络不是Damage Probability，NAR须看逐seed实测与纠错代价，不把拒绝所有改写包装为成功。

本轮两片已参与既往开发，仅诊断开发证据；三个seed为纯source训练重复，适应随机流和reference固定。使用既有Version类别内容对应假设，USA仅地理节点不解释为定量丢包。未开放WTT/AWF，无新源模型训练，无全矩阵扩展。主次候选及所有阴性保留。

18checkpoint与12控制状态均完整reference/query/valid重载；source BN底座锁定、独立混淆矩阵与聚类分配、reference-only聚类重拟合、120部署策略独立预测重算、历史rank策略精确复现及SHA核验通过。

本轮结论解释（冻结主判据评分后的只读整理）：去KL没有支持“移除直接约束就能增加纠错”的预测。它把Version/USA原始NAR降低到0.941/0.789%，但Recovery也从9.185/6.902%降到5.155/1.717%；同NAR1纠错均退化。这里的KL蒸馏对象是源邻居/几何混合软目标，不只是frozen分类器分布；当前消融不能判定所有teacher信息都是有害的，更不能宣称已消除残留的共模信息。

梯度平衡是本轮值得保留的局部信号，而非跨片成功：相对v6，Version/USA原始F1平均+0.370/+0.213pp，NAR +0.028/−0.137pp，Recovery +1.067/+0.232pp。同NAR1纠错+0.808/+0.084pp，只有Version达到预定+.5pp要求。Version三seed F1均增加，USA一seed−0.0036pp，不能宣称所有seed一致改善或统计显著。对应聚类纯度变化+0.318/+0.971pp，但共错集合仍只有3.636/1.890%被改正，且转移至其他错类的比例也增加；纯度上升不等于共模难例已经解决。

梯度平衡的几何倍率平均约32–39，六cell共20/2400步触发裁剪；不把“偶发裁剪”自动判为梯度爆炸，也不能沿用旧实验的零裁剪结论。有效倍率、加权梯度范数和夹角均逐步/按预定步数保存。

输出风险仍是独立瓶颈。固定v6的agreement_q95，Version/USA平均NAR约0.057/0.022%，Recovery只有0.615/0.330%，较rank-source1下降4.440/5.241pp，未通过保留纠错的判据。梯度平衡配rank-source1的USA NAR仍1.133%，因此原始候选更好不代表源阈值即可安全迁移。99%分位采用极少数改写，低风险不足以证明在同纠错率下更优。

后续若继续，应保留“几何与KL梯度相对失衡”这一待复核线索，同时把候选质量与验收阈值迁移分开评估；不要再把去KL、减少预测改写或聚类更纯直接当作安全纠错改善。本轮没有追加新arm、调倍率/分位阈值或进入外部评测。
