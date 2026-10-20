# 三痛点机制定位 v7

用户已明确恢复实验。本轮承接v2–v6及复盘反馈，属于已观察两切片诊断开发，不是新确认。以102类、18,553纯净源训练模型42/100/2026为唯一起点；不得变更原1024无标签reference/query隔离。所有目标标签仅评分/事后诊断，源valid标签只按冻结规则校准阈值和报告；源train仅已有原型和identity锚点。不新增源模型训练，不访问WTT/AWF。

一、BN因子分解：affine_only（v3既有TENT affine+原source统计）、stats_only（原affine+仅reference重新逐层校准population统计）、tent_both（既有TENT参数和reference统计精确复现）。另将既有v6 LoRA分别与stats_only和tent_both做无新优化的冻结组合。每个源seed/片5新推理状态共30；两种组合含历史双400步成本，不作为同预算联合优化成功证据。所有BN统计只来自reference，不使用query。source/frozen/LoRA为明确历史控制。

二、原型写入隔离与早期统计×LoRA：四arm×两片×三seed=24新适应，均400×64、Adam lr.00025、source identity32/步、r4 Q/V4096参数。memory_route仅以固定reference可靠性r调制prototype写入，所有样本仍可读原型/参与原动量加权梯度；memory_mass_control使用全batch同一个标量并匹配总weighted writer mass；stats_lora先校准reference BN统计再训练LoRA；stats_memory_route联合两因素。使用原纯净teacher logits/votes/full102目标，不让BN改变teacher；BN条件重提reference和同一1024源anchor的Token，identity参考仍是纯净源embedding。所有底座及校准BN在LoRA期间冻结。只有test-time LoRA参数训练，checkpoint固定final400。每arm模型/优化器/EMA重置；适应rng和批流20261019保持配对，源训练seed不是独立reference抽样重复。

三、评分拆解与reference归一化：固定v6主候选，不更新它；获取unlabeled reference的frozen/adapted预测/几何信号。Pearson/Spearman、方差与分量范围均报告。source1校准根据源valid真实标签、预定分位阈值网格满足NAR<=.01且更正>误改、至少10变化，按纠错最大/误改最少/阈值最大选取；不可行则全部回退并标记。raw combined、raw margin、reference-z margin、reference-ECDF margin、ECDF三分量均值融合分别比较。源valid规范化由其自己的changed pool、query规范化只由1024reference changed pool。若reference少于32变化，局部策略关闭。ref rank95是变化率规则，GMM2 tail是无监督分组启发式，均不宣称NAR保证；GMM只在BIC优于1分量且样本足够时启用，posterior不解释为安全概率。query禁止批统计、局部归一化或GMM拟合。

四、层级探针：256纯净source anchors与256reference固定前缀，无目标真标签。记录三个生成stage、time_projection和真实存在的两个Transformer block（本模型没有Block3）。unpaired source/reference仅算按source通道SD标准化的对角高斯channel-moment W2；CKA仅比较同一批reference的frozen/干预前后成对特征，不把unpaired CKA或分布差异称为不可逆信息丢失。第1/200/400步记录KL、pull、relation、identity各自梯度范数及夹角，不用query梯度。

评价：逐seed/片Accuracy、Macro-F1、ΔF1、NAR、Recovery；同一common-score Rec@NAR1/2为事后诊断。分离部署source阈值与标签定位曲线。主候选memory_route对matching mass control的两片F1各>=+.20pp且同NAR1不差才支持可靠写入；stats_lora相对v6 LoRA两片F1各>=+.20pp、raw NAR增幅<=.20pp才支持早期统计增益；评分主候选reference-rank-source1须两片实测NAR<=1%且Recovery较combined_source1各>=+.50pp。全部预定arm保留，不按最好seed重定主候选。不以GMM后验、CKA变化或oracle并集代替风险/纠错结果。

预算：GPU0/1/2分别seed42/100/2026，不干预既有其他进程；管线最多90min/4.5GPUh、3GiB、每适应900s。source-only预检通过后执行冻结批次，不依据目标反馈追加arm；失败仅允许记录并修实现错误。所有54新状态保存模型/LoRA、query/valid预测，全量独立重载、混淆矩阵与策略重算、BN作用域/源底座锁定及SHA审计。禁止全量扩展。
