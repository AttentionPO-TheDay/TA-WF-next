# GPU源期拟合诊断与官方DF配方参照

授权：2026-10-01用户明确“gpu有了，开始展开实验”，承接小集拟合检查及同预算原生基线建议。仅展开该诊断批次；不自动训练新结构或开启未来评价。

问题：当前生成器Transformer能否拟合小训练集？在相同源期样本与方向信息下，按官方结构/配方移植的DF是否明显优于当前模型？不预设过拟合或输入压缩是唯一原因。

## 数据、输入及权限

数据定位遵循 configs/datasets.json 的 proteus_temporal。本轮显式复用本项目 `20260927T032819Z_source_size_local_depth_factorial_dae714b7/artifacts/prepared.pt` 及 manifest.json（SHA-256在冻结清单）。仅 source150（102类×150=15300）和既有 valid（102类×5=510），不重新抽分验证集。缓存其他 source20/source80 不用于本轮训练。输入为既有前5000有符号时间戳取sign后的方向及padding；绝不解释为包大小。Transformer额外视图全部由相同方向派生，DF直接用方向序列。source标签仅CE训练和诊断；valid仅20次checkpoint选择及获准指标、预测核验，不梯度、不适应。Day14及其他未来、WTT/AWF均关闭。

原始数据不改写、不加载任何历史checkpoint。沿用已审核抽样行并保存本轮使用清单；运行前检查每类样本数与source/valid完整方向重复。固定valid已多轮观察，本轮不是独立确认。

## 实验顺序与预定裁决

1. 当前Transformer小集诊断，三个seed 21729/23407/22026；从source150既定顺序每类取前2条，共204条。关闭dropout及weight decay，AdamW lr=.001、batch64，至多3000步，每100步仅评价训练集；达到99%训练准确率停止。保留102输出类和原视图，不读valid用于该任务。三个seed均达到99%为拟合门槛通过；不通过只表示该预算下拟合失败，不能唯一归因于实现错误或信息压缩。
2. 正式DF参照三seed：下面审计的官方结构移植，方向5000包、Adamax lr=.002/betas=.9,.999/eps=1e-8/无weight decay、batch128、30轮，每轮重新打乱，保留不足128的末batch。验证轮次为1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30。官方原脚本使用最后一轮，故同时报告last；本项目best只在这20次按最高valid Macro-F1取最早，不能称官方原选模。
3. 当前Transformer三seed GPU复跑：1层局部编码、attention生成器、dropout .1、AdamW weight_decay .0001，batch64，12800步，1..6400 lr=.001、6401..9600 .0003、9601..12800 .0001。验证步与历史20次一致。所有权重从头初始化，GPU与CPU数值差异单独报告，不替代CPU旧结论。

主指标valid Macro-F1，辅以accuracy、训练拟合、last指标、步数、参数量、运行时间。DF相对Transformer三seed F1差均正且均值>=1pp才称本轮明确差距信号；不视为显著性检验。只一次训练样本抽样，三个seed是优化重复，不是三份独立数据。数据预算/输入信息/选模次数匹配，但原生训练配方使样本呈现量、步数、FLOPs不同；本轮评价整套流程，不能唯一归因于架构。禁止中途追加搜索或按valid继续延长。

## DF来源审计及移植边界

官方仓库 https://github.com/deep-fingerprinting/df ，固定commit `38df0c15a089f13228e4df06bd5269c8a37340fa`，下载源码保存在artifacts/upstream（仅只读审计，不导入执行）。依据 Model_NoDef.py 和 ClosedWorld_DF_NoDef.py。官方5000方向输入、4块32/64/128/256通道、每块两次kernel8/stride1/SAME卷积、SAME maxpool8/stride4、卷积dropout .1、512/512头dropout .7/.5，Adamax .002、30轮、batch128。

本地旧迁移DF与官方存在卷积偶数核padding、池化padding、bias和flatten尺寸差异，不能标为官方逐层等价。本轮新增run内df_reference.py，显式实现非对称SAME及20×256展平、channels-last flatten、bias、BN eps .001及momentum .01（对应Keras .99保留系数）、Xavier初始化。输出logits交CE等价于softmax交叉熵的目标。

本轮仍称“官方结构/配方的PyTorch移植参照”，不称原Keras精确复现：PyTorch与Keras随机数/初始化序列不同（本轮所有层随seed独立Xavier，未复刻官方Dense seed=0随机流），BN运行方差估计、Adamax底层计算有框架差异。类别数按本项目102适配，验证及标签权限按本协议。不会用官方论文不同数据的准确率作为本轮目标。

## 资源及核验

只用物理GPU0 RTX4090；GPU1/2已占用，不触碰。串行最多9个训练任务，CPU4线程、单任务45分钟上限、整批4小时上限，外层watchdog额外60秒退出宽限；不自动扩预算。全FP32、不AMP、不TF32，确定性算法，记录torch版本；保存抽样流、曲线、初始梯度、best/latest权重和最佳/末次预测。小集达到门槛提前停止；预算超时、非有限损失、数据/冻结哈希不符即停并保留已完成产物。

训练前检查SAME池化数值参考及4层长度1250/313/79/20、DF梯度、eval重复、当前模型CUDA梯度。训练后每任务重新创建模型重载best checkpoint复算预测，使用sklearn独立复算accuracy和102类Macro-F1。脚本、配置、计划、输入缓存、manifest及datasets.json哈希冻结于artifacts/freeze.json。使用旧项目的Python虚拟环境仅获取第三方依赖；sys.path只加本项目src，不导入旧训练代码。
