# 样本规模 × Transformer汇聚方式

用户授权CPU并行执行四组实验。准备阶段只做数据资格审计，不评分或训练；通过后冻结配置/代码/数据，后台执行。

## 数据与样本资格

保持已观察valid510条及原source2040条完全不变。原文件由configs/datasets.json定位，仅打开TemporalDrift/train.npz与valid.npz。扩大训练规模目标为102类每类80条，共8160；原source每类20条是其严格子集。按已有规则先float32再sign取前5000包方向。全valid2160条X仅用于重复排除，不扩大评分样本、不用其额外标签训练或选模。

训练候选须有限值、方向零值仅尾部padding、非空、标签为0..101整数；按实际输入方向散列去重。排除与完整valid重叠、标签冲突的训练方向组；原small样本优先保留，额外重复取最小原行号。原small若不满足资格则停止审计，不擅自换其样本。以固定sampling seed20260927在各类合格额外候选中选60条；任一类不足即停止并报告，不按评分换样本。保持原source行顺序作为expanded前2040行，便于共同训练子集诊断。保存全部计数、被排除行号、嵌套索引、原文件与prepared散列；源数据只读。

所有训练标签来自source。固定valid标签仅用于20次预定选模/开发评分；无未来日期、WTT/AWF、PCAP、预训练或TTA。扩展样本不是未见确认数据。训练集规模比较包含新增标签预算，应明确披露。

## 四组模型

A=每类20+当前CLS；B=每类20+更新后token masked mean；C=每类80+CLS；D=每类80+masked mean。

生成器仍为本项目LocalPacketGenerator(attention)，与当前HierarchicalViewTransformer联合从头训练。只修改三个视角local encoder的输出汇聚：CLS条件保持原路径；mean条件在单层Transformer计算后，取普通token输出的有效mask均值，排除CLS与padding，空视角置0，再沿用原全局模块。不改变输入CLS参与attention、位置编码、宽度、层数或参数数目；用明确的run-local forward hook实现，checkpoint重载时根据配置重建hook，原代码不变。生成器仍接受分类损失反馈；不叠加生成器锚定或强分类器正则化。

这项对照同时改变聚合规则与是否使用普通token的上下文化输出，不能把收益唯一归因于token关系建模。首先验证当前CLS实现复现、mask排除、空视角、梯度通路和参数初始化匹配。

## 配对与优化预算

三模型初始化seeds21729/23407/22026，四组同seed初始权重完全一致；同数据规模两种汇聚共享batch索引顺序与初始随机流。新seed避免重复已有模型训练，不能当作新验证样本。

每job固定3200次AdamW更新，batch64全批，共204800个样本呈现；学习率0.001、weight_decay0.0001、dropout0.1、source CE，无scheduler。用连续的随机排列流组成batch，跨排列边界补足64，不丢尾样本；small/large平均访问次数约100.39/25.10，明确不是相同epoch数。100个记录block，每block32步；每5block评分，共20次，最高valid Macro-F1、并列最早。比较相同步数的效果，不能把较大训练集尚未充分收敛解释为数据无用，也不根据结果延长。

各评价点记录整个本组source、共同small训练子集、valid的accuracy/F1；跨样本规模解释训练拟合时优先用共同small指标，避免比较不同样本组成造成误导。保存best/latest、优化器/RNG/索引游标和访问次数，预测集为source/common_source/valid。

## 判据与报告

预定比较B−A、C−A、D−C、D−B、D−A。各项候选门槛：valid F1均值至少+1pp、三seed均正、accuracy均值不降。全部报告，不按valid事后挑seed/强度。另逐seed报告样本主效应=((C−A)+(D−B))/2、汇聚主效应=((B−A)+(D−C))/2及交互D−C−B+A；均为描述性开发证据。每类仅5个valid样本、只有一次嵌套训练抽样，不声称独立确认、统计显著或唯一因果。

## 资源与核验

12job，CPU6worker×2线程。每job5400秒加180秒收尾，核验每job600秒，整轮11400秒上限；单worker RSS4GiB，总24GiB。约束是时间上限，不是必须耗尽预算；超时/错误保留不完整结果，不自动重启、增加步数或替换数据。启动检查后不持续轮询，用户询问时查看。完成后自动汇总。

显式复用本项目上一轮common/监督/核验模式和src，无旧项目训练代码或checkpoint依赖；仅使用/home/rbf/TA-WF/.venv环境。数据准备有界，只产生本run数据产物。正式运行前冻结输入与实现散列；独立进程复算12job×3角色=36组预测、指标、选模和步数、样本流权限/访问次数、generator更新、配对初始化。完整通过才标completed。
