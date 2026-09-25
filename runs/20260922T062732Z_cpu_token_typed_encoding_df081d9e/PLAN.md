# 20260922T062732Z_cpu_token_typed_encoding_df081d9e

问题：相同token骨干下分桶加连续长度专属编码能否改善分类学习

状态：frozen；训练前冻结。用户授权CPU下一轮实验。

数据：复用本项目20260922T060351Z_cpu_token_native_classifier_6463ceb7已封存的source2040/valid510 token缓存和原抽样清单，不重新划分。原source已排除完整valid重叠并去重。本轮核对缓存hash、source/valid隔离、实际生成字段及标签。共享只读原始路径通过configs/datasets.json定位。source标签用于梯度，valid仅每轮选模/开发评分；未来日期、WTT/AWF关闭，无适应。缓存复用不当独立数据重复。

四条件：continuous长度Linear(1,16)；bucket长度Embedding(14,16)；hybrid为两者相加；hybrid_norm在hybrid的方向/长度concat+位置之后增加LayerNorm(32)。方向Embedding(3,16)，连续长度log1p(count)/log(5001)，桶floor(log2(count))+1，0为padding。其余保持32维、两层k5保序Conv1d、逐层mask、16段masked mean、Linear512-102。方向/长度各自编码后融合；不添加窗口、边界、资源、时间或packet旁路。仅借鉴CipherSight数值双编码，不是其求和式全字段tokenizer、TLS层级或语义监督复现：https://arxiv.org/html/2608.13905v1 。实现沿用本项目上轮TokenNet/训练流程，显式重新实现；无旧目录代码或checkpoint依赖，仅用旧虚拟环境依赖。

主比较hybrid-continuous：相同精确信息可用性，桶是精确长度的确定性派生；次比较hybrid-bucket有额外精确信息，不能单独归因架构；hybrid_norm-hybrid隔离归一化。所有公共模块采用独立固定初始化seed，确保各条件公共初始权重相同；同seed相同batch顺序。该配对初始化不同于历史脚本，不要求历史数值完全复现。参数差应小于1%，记录实际参数而非宣称严格等容量。

预算：4条件×seeds1729/3407/2026，15epochs，batch128，AdamW lr0.001 wd0.0001，CPU4线程，GPU隐藏，总timeout600秒。全部从头训练。valid macro-F1最大选epoch，平局最早；指标accuracy/macro-F1、训练动态、参数/耗时。禁止观察结果后加seed/epoch/条件。非有限、隔离失败、hash不符或超时停止并保留失败。

预定筛查门槛：hybrid相对continuous三seed的macro-F1差均>0，平均accuracy差>0；hybrid相对bucket同样门槛另报，未通过不称额外精确分支稳定有效。norm单列报告，不事后更换主比较。不作显著性/抗漂移/独立确认结论。valid已经多次用于开发。

检查：padding不影响输出、分支梯度、连续分支关闭时精确值不可见、公共初始化相同、桶/连续生成一致、checkpoint重载、全部预测指标独立重算、选模和封存核验。保存配置、代码hash、输入、预测、历史、模型。此轮token encoder训练/评价都保留，仅评估其学习能力；训练期专用分支最终蒸馏到部署模型仍未实现。
