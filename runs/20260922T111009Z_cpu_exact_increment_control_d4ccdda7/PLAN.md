# 20260922T111009Z_cpu_exact_increment_control_d4ccdda7

问题：匹配专属层后精确长度是否优于仅桶信息及常量分支

状态：frozen；访问训练数据前冻结。用户授权继续CPU验证。

问题：上轮独立处理/固定衰减的收益，究竟需要精确长度，还是仅增加处理层及分支就足够？固定source2040/valid510原清单，不重划。来源20260922T060351Z_cpu_token_native_classifier_6463ceb7封存token，source已排除完整valid方向重叠并去重。原始数据根由configs/datasets.json解析，只读；source标签梯度训练，valid仅选模及开发评分，未来/WTT/AWF关闭，不适应。

四条件全部使用方向Embedding3-16、长度桶Embedding14-16、桶专属残差处理B+GELU(Linear16-16(B))、位置Embedding512-32、两层32通道k5卷积、mask、16段池化、Linear512-102。bucket_mlp仅桶处理；fixed_exact增加与上轮相同的连续分支C=Linear1-16(log1p(count)/log(5001))，融合0.1*(C+GELU(Linear16-16(C)))；fixed_bucket与fixed_exact完全相同结构参数/初始化，但连续分支输入只用桶下界log1p(2^(bin))/log(5001)，不读取精确值，padding为0；fixed_bias同结构但该分支输入全0，只保留可学习偏置/后续处理。全部从头训练，非训练后关闭。

观察预算5000、512run，字段及padding与前轮相同。fixed_exact-fixed_bucket主比较匹配全部参数与处理，仅改变桶内精确信息可用性；桶下界是预定无拟合代理，阴性不等于精确信息永远无用。fixed_exact-bucket_mlp补直接缺失的对照；fixed_exact-fixed_bias区别输入相关分量与常量分支；fixed_bucket/bias相对bucket_mlp用于解释结构收益。fixed_bias的连续权重输入恒零因此无数据梯度，不能称有效容量完全匹配。各条件公共模块配对初始化，同seed同batch顺序。fixed_exact复跑上轮separate_fixed为同轮锚点，不算独立新数据证据。

预算：4条件×seeds1729/3407/2026，15epochs，batch128，AdamW lr0.001 wd0.0001，CPU3线程，GPU隐藏，timeout600秒。每轮valid macro-F1最大选checkpoint，平局最早。非有限/hash错误/隔离错误/超时停止。不得看结果后加预算或调比例。

预注册主门槛：fixed_exact-fixed_bucket三seed F1差均>0且平均accuracy差>0，同时固定报告对bucket_mlp和bias的同门槛。失败则不得称精确长度增量稳定。报告均值/SD、全部逐seed配对差、训练动态、参数、耗时；不作独立确认/显著性/抗漂移结论。

验证：精确值改变对三个非exact条件输出无影响；exact在桶代理输入下与fixed_bucket初始输出完全一致；桶/公共模块初始化一致；padding、分支梯度、bias输入零梯度符合预期；全量生成器重建token、缓存/原始方向hash、隔离、checkpoint重载、选模、预测指标独立重算。配置代码输入封存保存，所有产物归本run。实现显式复制本项目20260922T064218Z_cpu_token_gated_fusion_24ddd51d脚本后最小修改，不依赖旧目录训练代码/checkpoint，仅旧虚拟环境依赖。训练/评价都保留token，不验证训练专用蒸馏。valid已多轮开发，不是全新确认。
