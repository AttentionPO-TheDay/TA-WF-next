# 逐级生成器容量×BatchNorm及Mixup并行两线路 v1

用户授权“两条线路都做，并行设计”。解释为容量×BatchNorm与独立Mixup线路，基线为当前progressive_transformer86.536%，统一实验批次12新训练+3历史baseline，不叠加未测试的宽/BN与Mixup组合。停止跨头冻结复用优先项，当前目标提高源验证accuracy，90%三seed均值单列。

条件：baseline通道32/64/128、无BN/无Mixup（3历史）；wide通道64/128/256；bn原通道+BN；wide_bn加宽+BN；mixup原通道无BN+Mixup alpha.2。4新条件×21729/23407/22026。调度按每seed bn/mixup/wide/wide_bn交错，GPU0最多3路，让两线同时推进；无CPU训练、其他GPU不使用。

生成器原三stage、每stage两kernel5 padding2卷积/1x1skip残差/GELU、stage后平均池化3/5/1不变，输出通道128或256投影到80维，与固定30维完整log-count拼接120×110。Transformer及投影/位置/读出完全固定。同width有/无BN卷积初始化逐元素相同；所有条件分类器初始化一致。BN插在各stage的conv1之后/GELU之前及conv2之后/残差相加之前，6个BatchNorm1d，eps1e-5/momentum.1/affine/track_running_stats，gamma1,beta0,mean0,var1；skip和输出不加BN。保留卷积bias以维持配对初始化，BN前bias梯度可近零，不以所有bias均更新作为有效性条件。BN训练沿source batch×time统计，验证和source评价均eval、buffer不变；best counter必须等于best训练step，不允许valid更新统计或重校准。

Mixup完全沿用已有单独测试规则：source span_mask后、log1p前混合原TAM；每batch Beta(.2,.2)权重和独立配对，标签CE相同权重。NumPy default_rng(seed+80000)与torch/dropout/mask隔离。时间戳分支禁用，eval不增强。该条件只在已支持的新生成器基线上测，不组合BN/加宽，不做alpha扫描。

数据configs/datasets.json定位，复制上轮冻结prepared source15300/valid510缓存，相同行/标签/输入不重划；source标签训练、valid仅固定20次选模/指标。未来/WTT/AWF/适应/teacher均关闭。全部新任务从头初始化，历史checkpoint仅重载审计。batch64/12800steps、CE、AdamW lr.001 wd.0001，6401降.0003/9601降.0001，randperm(seed+9000+cycle)，span_mask seed+50000、p.5、连续90bin同方向。eval steps3680+480i共20，accuracy最早严格最大best，完整保存last。float32无AMP/TF32，确定性，2CPU线程/worker。

比较4候选对baseline；wide_bn对wide/bn；交互(wide_bn-wide)-(bn-baseline)，逐seed及均值accuracy/F1/source/valid/参数/时间/显存完整报告。开发门槛accuracy平均+1pp、3/3正且F1不降，90%另判。BN和容量影响优化/表达，不预先断言能解决过拟合；同验证集多轮开发，非独立泛化/通用性确认。

预算GPU0最多3路、单任务3600秒、整批18000秒；预检3×最宽BN的显存保守估算含优化器状态/上下文与4GiB余量，超预算须在正式冻结前限并发，不改batch。先核验6基线预测/初始化、宽度配对、BN train更新与eval不变/Mixup对应/有限梯度/固定计数及时间戳禁用；正式训练前冻结code/config/PLAN/SOURCES/data/preflight/reference hash。非有限/权限/显存/梯度/重载/选模错误立即停止并保留部分产物，不追加预算。
