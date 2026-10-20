# 候选纠错与验收 v3

## 依据与目标
v2诊断与外部v4均已反馈到方案；本轮是开发证据，不是新确认。v2训练熵门控错过几乎全部纠错；冻结Top5保留较多正确类别。检验(1)跨候选类别的软监督能否优于硬教师；(2)独立源邻域证据能否改进软目标；(3)按修改证据验收能否改善NAR/Recovery。所有新组合及阴性结果完整报告，不根据目标结果调参。

## 复用与信息权限
唯一底座是18,553条Temporal纯源训练得到的三seed模型。复用v2三份128维源表征/GMM1、源锚索引；不训练底座。复用诊断v1十个slice缓存与同一query/reference、隔离审计和内容支持102类身份假设。Version非盲测、Network仅地域、Behavior仅URL分组；不开放新WTT/AWF评分。
source train labels只建立源邻居投票和锚点，source valid 2160行标签仅校准修改验收阈值，不进入adapter梯度或目标分布；valid是已用于底座选模的验证集，不称独立校准确认。target reference无标签1024行用于400步优化；query标签仅终态评分及事后分析。query输入可做固定单样本分类与源邻居检索，不拟合query统计；标准TENT的query批统计另行明确标为transductive，严格reference版不读query统计。

## 冻结矩阵与预算
三个target: hard / soft_teacher / soft_neighbor，分别配Residual3866与QV rank4 LoRA4096，六种适应每seed×10slice；同源初始化、索引流、400步、batch64、source32、Adam0.00025、clip1、无dropout，固定最终步，无query选模。新增TENT同预算仅BN affine；它有两种推理状态，不冒充独立训练重复。frozen逐行重算。共210个训练任务（180 adapters+30 TENT），270 raw状态评分（frozen+2TENT+6adapter），每个adapter另评valid_entropy与evidence_accept两种固定验收，共630个评分状态。GPU0/1/2一seed各自并行；上限2小时/6GPU小时、单元600秒、输出5GiB，错误/不有限/超预算停止保留。历史v2硬原型残差/LoRA只作明确复用对照，其损失无本轮KL，不能将差别单独归因软硬标签。

## 目标与共同损失
单原型标准化128维空间沿用v2。候选集合为冻结教师Top5（hard为argmax onehot）。soft_teacher在Top5内使用T2 softmax；soft_neighbor额外乘以源邻居类别投票，再归一化。邻居是标准化源向量归一化后的余弦近邻，按source row SHA hash parity分两个固定不交叠bank，各取10邻居；带宽来自source valid最近邻距离的中位数，源投票平滑0.01。所有q及分组由固定教师/源库生成，目标标签不参与。共同损失 = KL(q || softmax(logits/T2))*T2² + .1*expected prototype squared distance + 1*类间余弦关系 + 1*source锚表征恒等。hard/soft/neighbor匹配损失，仅q不同。所有base参数及BN固定，只有adapter更新。

## 修改验收
raw全部采用candidate。valid_entropy沿用基于H/D的逻辑，但H95与D99均从source valid而非训练拟合行获得。evidence_accept不按低熵拒绝：只有candidate预测发生变化才检验candidate在冻结Top5中、两源bank对candidate相比frozen类别的最小log-vote增益达到阈值，并且距离不超过source valid D99。相同预测的logits也统一回退frozen，便于逐行证明。阈值在source valid上以源邻居建议的修改作校准，固定grid[0,.25,.5,1,2,4]由小到大取首个满足>=10修改、纠错:误杀>=2、NAR<=1%的阈值；无可行值则全部拒绝并诚实报告。不同target/adapter共享每seed同一阈值，不用目标表现选择。query不接受任何更新，拒绝行logits必须逐位等于frozen。该规则不保证目标零误杀。

## 检查、汇总与停止
纯源预检先证明q概率和、候选约束、跨类别切换资格、0初始化同模式逐位一致、梯度有用、底座不变；所有源校准/代码/输入SHA在目标评分前冻结。全部raw checkpoint重载、630状态指标独立复算及旁路核验。输出逐seed/slice F1、Accuracy、ΔF1、NAR、Recovery、接受/拒绝/修改计数、校准可行性与成本；三因素分别等权后再三因素等权，三个seed均值±样本SD；报告soft_neighbor-soft_teacher及soft_teacher-hard配对差，另比较raw/验收取舍。主开发门槛：相对hard同adapter三因素平均F1>=+1pp且NAR不升，或保留raw至少80%纠错同时NAR减少至少30%；与reference-only TENT同预算比较。若均不满足则不视为机制成功，不继续扩大参数搜索。
