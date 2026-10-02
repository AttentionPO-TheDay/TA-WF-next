# 网站指纹生成器跨分类头结构适配 v1

用户明确网站指纹领域通用生成器，并授权开始。此轮验证统一token接口和生成器结构跨联合训练分类头的适配性；只在已观察TemporalDrift源期，不能称冻结生成器权重迁移、方向-only/缺时间适配、跨数据集通用或抗漂移已成立。那些阶段待方法冻结及独立权限/预算。

A shallow_transformer原基线3seed历史复用；B shallow_mlp、C progressive_transformer、D progressive_mlp、E rf_matched各3新，共12新训练。seeds21729/23407/22026。A最终来源span_mask历史run，所有来源/初始化/预测注明；E官方RF代码显式复制，从头初始化，不是原创生成器或蒸馏。原生RF85.948%仅历史配方参考。

A/B生成器原共享dilation1/3/9、kernel5两层2→16→16，log1p及保序子区间特征。C/D三stage通道32/64/128；每stage两个kernel5 padding2 stride1普通Conv1d，GELU(conv2(GELU(conv1(x)))+Conv1x1skip(x))；全bias，默认PyTorch初始化，无新增BN/LN/生成器dropout；第1/2stage后保序3/5bin均值，网格1800→600→120；最后Conv1x1 128→80不加activation，120×80学习特征与原固定30维log-count拼接，120×110接口相同。残差/通道/连续局部模式/网格为完整设计包，参数/FLOPs非等价，不能唯一归因于深度或层级。原计数固定分支完整保留。

A/C分类头原2层Transformer d128/4heads/FF256 dropout.1。B/D沿用已存在TokenMLP：两逐token残差LN128→Linear512→GELU→Dropout.1→Linear128→Dropout.1，然后原finalLN/分视角masked mean/linear256→102；MLP无跨token交互。四条件保留相同投影/位置/类型接口，packet100槽位全mask及分支参数冻结。生成器不接收标签/类别数；分类器类别数可配置。条件共享形状且功能相同的初始化按seed一致；C/D生成器逐元素一致，B/D头一致，不把不同结构同seed误称同初始化。各条件联合训练，不是同一已训练生成器跨头迁移。

E官方RF102原BN/ReLU/pool/dropout结构不变，输入与A-D同log1p(TAM)、same source span_mask，统一AdamW/batch64/步数/索引/20验证机会；和原生raw-TAM/Adam/batch200/30epoch属于不同完整配方。E用于诊断结构×配方的差异，不从它与原生差距唯一归因某个超参数，也不称完全等计算。

数据通过configs/datasets.json定位，复制上轮冻结prepared缓存，source15300/valid510固定行不重划。source标签用于CE；valid仅20次选模/指标；未来/WTT/AWF/适应/teacher关闭。所有新任务从头训练。batch64、12800步，AdamW .001/wd.0001，6401降.0003、9601降.0001。索引seed+9000+cycle；source mask p.5/连续90bin双方向同步/seed+50000。评估无增强；20步3680+480i，最大valid accuracy最早严格改善，另存last。固定float32无AMP/TF32/确定性。

比较C−A、D−B及交互(C−A)−(D−B)检查新生成器跨头效用；B−A、C−D检查头贡献；D−A整体候选，E−A/E−C配方内结构参照。主开发门槛平均accuracy+1pp、3/3seed正且F1均值不降；90%独立判断。跨头适配候选须C−A和D−B都通过，不以单一胜者声称通用。所有条件逐seed/参数/时间/显存/source/valid/best/last完整报告。

GPU0最多3路、其中RF最多1实例，其他GPU不使用；每任务2CPU线程、3600秒，整批18000秒；执行前用真实source64前后向/优化器状态内存检查，计三份数据/上下文与4GiB余量。失败保留不自动增预算。冻结前验证baseline重载source/valid，固定计数/接口/时间戳禁用/类别数无关、同模块初始化、非零有限梯度，RF wrapper与原实现log-input一致；预检不训练选模。代码/config/PLAN/SOURCES/data/reference hashes全部冻结。
