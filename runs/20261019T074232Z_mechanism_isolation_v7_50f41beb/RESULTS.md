# 三痛点机制定位 v7 结果

**实验已完成；三个预注册跨切片改进判据均未通过。** GPU0/1/2完成24新适应、30推理分解状态，54个状态均完整重载query/valid并复核。所有结果保留，没有追加参数搜索。

[全部状态与三seed标准差](table_main.md)；[真实部署评分策略](table_scoring.md)；[风险—纠错前沿](figure_frontier.png)；[层级探针](figure_layers.png)。

一、可靠原型写入没有解决共模偏差。memory_route相对写入总量对照的F1变化，Version为−0.117pp、USA为+0.036pp；同NAR1纠错变化为−0.304/+0.669pp，未达两片一致改善。它确实缩小了原型移动，但较小移动不等于较好纠错。可靠性仍来自同一编码器，reference真标签没有进入筛选或品质验证。

二、BN统计重估不是通用修复。stats_only的ΔF1为−0.240/−1.233pp、NAR6.560/7.946%；stats_lora相对现有v6 LoRA的F1变化−0.917/+0.057pp，NAR反而增加1.852/2.589pp。源/目标通道距离略降并未转化为安全改善。

BN affine与统计有强交互：USA affine_only ΔF1−2.349pp（SD5.805pp）、stats_only−1.233pp，而tent_both +2.212pp。affine来自已在批统计下优化的TENT，移到source统计可能失配；因此单独移植失败不能证明affine优化本身无效。历史TENT与LoRA组合达到+4.660/+3.406pp，但NAR4.757/5.779%，且含历史双400步成本。其common-score Rec@NAR1为4.154/4.503%，v6为4.402/2.810%，仍非两片同时改善。

三、reference相对评分有局部价值，但风险不能从源域自动转移。margin_ref_rank_source1在Version ΔF1+1.808pp、NAR0.710%、Recovery5.055%，比combined_source1多纠错3.182pp；USA ΔF1+1.610pp、NAR1.450%、Recovery5.572%，NAR不达1%。USA三seed NAR1.439/2.122/0.787%，不能用均值掩盖风险。rank95和GMM尾部虽然低NAR，两片Recovery均不足1%；安全来自极少改写，不代表同纠错率风险更低。

四、梯度探针指向主要目标仍由旧teacher驱动。第400步memory_route的加权KL梯度范数约0.411/0.390，pull约0.0046/0.0042、relation约0.0149/0.0122；KL明显主导。stats_lora在USA的KL/identity梯度cosine到−0.179，说明部分更新与源identity保护相冲突。本轮9600步均未触发梯度裁剪，有限探针不能排除所有数值或优化问题。

融合诊断需统一候选口径。scoring.py原始correlations报告所有changed样本，其中也包括wrong→other-wrong，AUROC为“真正改对 vs 其余改写”。本补充同时输出仅recovery/damage的相关矩阵与AUROC，以便与旧复盘同口径比较；不修改任何冻结评分、阈值或主判据。

version_045：在recovery/damage池中combined/margin AUROC=0.4945/0.6853；combined与margin Pearson/Spearman=-0.026/0.091。全部changed池中gain sigmoid≥.99占比64.2%，源距离排除8.2%。

network_USA：在recovery/damage池中combined/margin AUROC=0.5132/0.6635；combined与margin Pearson/Spearman=0.012/0.031。全部changed池中gain sigmoid≥.99占比52.7%，源距离排除10.1%。

下一步应先隔离teacher目标本身及BN affine×统计协同，而不是继续加原型/门控模块。可在另行冻结的小预算中，对固定旧teacher KL的保留/移除作严格匹配消融，并保留source约束、目标几何损失与同风险指标；这只是待检验假设，当前尚未执行。评分方面需单独解释源valid→reference→query的风险转移，不能用无监督GMM后验冒充误改概率。

实验边界：只用已观察的Version045/USA、三个18,553纯净源模型；reference1024行无标签，query未用于任何BN/原型/归一化/GMM拟合。所有适应400步，未选目标checkpoint。source-valid标签仅按冻结规则选阈值和终端评分。三seed为源训练重复、reference与适应随机流固定，SD不代表新切片置信区间。Version采用既有内容支持类别假设；Network仅地域节点，不宣称定量丢包或时延。未进行WTT/AWF新评分，也未扩展全矩阵。

层探针source/reference各256行；unpaired只使用channel moment W2，CKA只对同一reference的前后配对，不证明不可逆损坏。W2按各干预source通道SD标准化，需结合预测指标解读。原型总量对照匹配自身每批weighted writer mass；因状态演化，不能保证跨臂每一步总量完全相同。

冻结主实验代码/config/输入SHA未改；interpret.py为主实验完成后的只读诊断补充，未新增训练或选模。54状态完整query/valid重载、独立混淆矩阵、42部署策略和源阈值复算、BN范围与源底座锁定均通过。
