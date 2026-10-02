# 软目标与Mixup v1

用户授权发散思路开始新实验。依据：真实RF蒸馏未过门槛，而打乱目标78.824%过数值候选门槛；不把随机教师映射作为最终机制，先检验简化正则化，同时探索TAM输入混合增强。

固定原77.647%学生生成器+Transformer、log1p、两层16通道共享尺度1/3/9、原AdamW及source span_mask，无未通过的归一化/层级模块。4新条件：均匀软目标、同seed源RF概率边际、Mixup、Mixup+均匀软目标。每种21729/23407/22026，共12新任务；CE及打乱教师各3seed明确历史复用。比较四新条件对CE；前两对打乱诊断目标复杂度，组合对两个单独条件诊断交互。全部报告，不追加调参。

软目标：T2/lambda.5，hard/mixed CE+.5*T² KL(q||student)，batchmean。uniform q=1/102；marginal q=mean_source softmax(RF logits/T)，非mean logits。边际数据仅source15300，不读valid logits，历史教师曾按valid选模，三位固定同seed，其历史训练额外成本单列。其余新条件不加载教师checkpoint或soft targets，无教师推理依赖。学生均从头初始化。

Mixup：alpha.2，每batch一个Beta权重，独立随机batch行配对；先按既有规则分别遮挡每行，再原始非负TAM线性混合，之后模型log1p。两方向同一配对/权重，标签CE同权重；不是物理新trace或保证真实流量分布。时间戳输入保持原行但对应分支已mask/frozen，不能影响logits。Mixup RNG NumPy default_rng(seed+80000)，与样本索引/遮挡/torch dropout分离。eval使用clean固定样本，无mixup或遮挡。

固定TemporalDrift source15300/valid510，configs/datasets.json定位只读数据、复用冻结cache不重划。source标签可梯度，valid仅20次选模/指标，future/WTT/AWF/适应关闭。batch64/12800步，3680起每480步20次，最高valid accuracy最早严格改善。原AdamW lr.001/.0003/.0001，seed+9000+cycle索引，seed+50000遮挡，各条件同seed初始参数与索引一致。

门槛：对CE平均accuracy≥1pp增益、3/3seed正且F1不下降；90%另判。优化seed不是独立数据重复，现valid已反复开发，不声称独立泛化/抗漂移。

预算GPU0最多3路并发，2CPU线程，float32无AMP/TF32；每任务3600秒、总18000秒。CPU仅准备/审计。执行前冻结代码/配置/数据/历史文件hash；测试Mixup配对/标签权重、概率平均及KL T²、detached目标、eval不增强、旧学生预测重载和新初始化；3条件source64反传/资源检查。完成后预测独立指标/初始化索引选模审计，失败保留不自动加预算。
