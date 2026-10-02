# 生成器更新约束 × 分类器正则化

状态：执行前冻结。用户授权四组正则化实验；CPU后台执行，启动核验后不持续轮询，用户询问时查看结果。新run唯一目录。

## 问题与既有证据

固定生成器后重训实验已完成：训练后生成器对应source accuracy77.55%、valid26.27%；随机生成器对应61.24%/26.08%。整体训练/验证差距明显，但尚不能唯一归因于生成器或分类器。该实验不重复，本轮从头联合训练，检验限制训练集拟合是否带来可泛化收益。

## 四条件与算法

R00：基线，分类器dropout0.1、AdamW weight_decay0.0001，生成器无锚定惩罚。
R01：只加强分类器正则化，dropout0.3、weight_decay0.01，生成器无锚定惩罚。
R10：只增加生成器初始化锚定惩罚，分类器沿用基线。
R11：同时启用R01和R10。

全部使用当前LocalPacketGenerator(attention)+HierarchicalViewTransformer，119578参数，生成器6064、分类器113514；两部分都由source分类损失端到端更新。dropout同时覆盖分类器显式Dropout层和MultiheadAttention概率；生成器本身无dropout。分类器正则化作为一个组合干预，不能区分dropout与weight_decay单项作用。生成器AdamW lr0.001、weight_decay0.0001，各条件相同；分类器lr0.001。无scheduler、增强或其他新增机制。

生成器约束精确定义：L = mean(source CE) + lambda/2 * sum_{所有generator参数元素}(theta-theta_initial)^2。R00/R01的lambda=0，R10/R11的lambda=0.01；不按参数数目或batch大小再归一化。theta_initial是本job从头初始化的常量副本，涵盖卷积、汇聚打分与投影及bias；无梯度、不使用历史checkpoint。初始化时惩罚为0；每步记录总损失/CE/惩罚，评价点记录参数偏移与惩罚，独立重算。系数固定为一项有界开发假设，不依据valid校准或搜索，不声称最优。

## 数据、配对与预算

仅TemporalDrift固定source2040/valid510、102类每类20/5条、前5000包方向；prepared.pt、manifest散列及configs/datasets.json指向共享只读数据。source仅用于训练/诊断，valid仅按预定规则选模/评价；不重新划分，不开放未来日期、WTT/AWF或PCAP。不使用预训练/TTA/蒸馏。

训练seeds11729/13407/12026，均为新的模型随机初始化，以避免重复已完成的相同配置；不是新数据或独立确认样本。每个seed四条件共享模型初始state、batch顺序、初始随机流，dropout概率的干预本身允许掩码差异。新baseline是同轮新seed对照，历史结果只作研究背景，不当本轮独立重复。未加载任何旧模型权重。

每job100epochs、batch64；每5epoch评价一次，共20个选模机会，以最高valid Macro-F1、并列最早选best。训练/验证accuracy与Macro-F1均报告，明确区分在线带dropout训练准确率与eval模式source准确率。保存best/latest、优化器/RNG、逐轮日志、预测/配置/审计。

12个训练job，CPU6workers×2线程，最多12线程；每job3600秒训练上限，加监督器180秒收尾宽限，每次独立核验300秒；全pipeline7800秒上限。单worker RSS4GiB、总24GiB。沿用既有同架构联合训练的可行性证据，合成检查验证本轮新损失与优化分组，不拿真实验证分数选参数。达到预算或错误保留部分结果、标记不完整，不自动重试/加seed/延长。

## 预定比较与判据

主比较R01−R00、R10−R00、R11−R00；补充R11−R01和R11−R10。每项预定候选门槛：valid Macro-F1均值增加≥1pp、三seed差均正、accuracy均值不下降。任何条件未过门槛均保留阴性/不稳健；不只展示最高值，不事后更改强度。全体原始seed指标、source-valid差距、参数/token变化一并报告；训练准确率或差距下降但valid不升不算成功。

逐seed报告2×2主效应：生成器=((R10−R00)+(R11−R01))/2，分类器=((R01−R00)+(R11−R10))/2，交互=R11−R10−R01+R00。作为描述性效应，不以三个共享valid的seed作显著性或唯一因果证明。若约束未降低相应拟合/参数偏移，须说明操纵效果有限；阴性仅针对此强度与预算，不否定所有正则化。

## 实现来源、核验与限制

训练/核验/监督脚本显式复制并最小修改自本项目20260926T040212Z_classifier_generator_parallel_946c81c4，无旧项目代码运行时依赖。只显式复用/home/rbf/TA-WF/.venv环境。执行前审计固定行索引、标签计数、原始方向/标签散列与source-valid重叠；合成测试验证初始化配对、锚定梯度只作用generator、lambda=0精确回退、非零约束朝向初始化、classifier正则分组/attention dropout及恢复一致性。配置与代码散列冻结后训练。

完成后独立进程重载12个best模型、复算24组source/valid预测及指标，核对选模历史、初始化、优化器分组、生成器反馈/偏移与惩罚。全部核验通过才标completed。TemporalDrift valid已被多轮用于方法开发，不能作为未经反馈的泛化确认；当前只检验源期分类能力，不宣称抗时间漂移。
