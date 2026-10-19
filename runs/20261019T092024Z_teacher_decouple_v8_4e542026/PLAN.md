# Teacher目标解耦与reference风险路由 v8

承接v7诊断反馈的有界开发，用户明确授权发布执行。不是未见确认。source纯净18,553行/102类、三个源训练seed42/100/2026；Version045/USA沿用不可变1024无标签reference与独立query。没有新source训练，不访问WTT/AWF。仅source/valid标签按冻结配置使用；目标reference标签不可用，query标签只作终端指标、纯度/错误迁移及事后风险曲线，不进入模型、记忆、阈值或聚类拟合。

选择纯后端路线：原始source BN affine及running统计均完全锁定。仅Transformer Q/V秩4 LoRA4096参数更新。历史v6 sharp_target_only为匹配控制，SHA及预测核验，不再训练已完成控制。三arm×两片×三seed=18新适应，均400步、batch64、source identity32、Adam .00025、clip1、相同初始化和样本流20261019；checkpoint固定final400。保留原动量权重、固定source pull/relation锚点、动态teacher原型目标、identity和feature/probability memory更新。只改变KL直接项及C组几何整体倍率。

A no_teacher_kl：KL系数0，.1pull+relation+identity保持原样。B confident_teacher_kl：只对原始teacher H<=source-valid H75且head==原始最近源原型的reference样本施加KL；全batch均值，不重新归一到mask计数。C geometry_grad_balance：保留完整KL，以当前LoRA参数上KL和(.1pull+relation)的梯度范数EMA(.9)比值决定lambda，截断[1,50]，geom梯度<=1e-10时lambda1；倍率detach，不对倍率反传，identity保持1。记录各步范数/倍率及第1/200/400步梯度夹角。不是完整GradNorm算法，也没有额外优化目标或trainable loss weights。

重要识别边界：qfixed仍由旧teacher及source邻居构建，几何target仍含其软标签。本实验隔离直接KL是否阻碍纠错，不把A称为完全teacher-free。B的几何一致性同样非独立证据。本轮不试BN拼接，不进行协同TENT/identity梯度投影。

诊断：每方法提取reference和query embedding；标准化保持source固定。只在reference上KMeans102(n_init3,max_iter100,random_state20261019)，query分配固定最近中心，query标签仅计算多数类purity/ARI/NMI，不能用多数类映射输出预测。固定共错subset=冻结预测错且head=source nearest=原source双bank投票；报告其纠错、仍同错、转移其他错的条件比例。对照frozen/v6亦按同一reference聚类规则处理，保留所有seed。query聚类纯度是标签诊断，无部署选模资格。

主终点为A对固定v6控制：两片各平均Rec@NAR1提升>=.5pp、rawF1不降、rawNAR增加<=.2pp；B/C同判据作为次候选，不取代A。共同曲线仅使用无sigmoid的normalized_margin_gain，全部changed同候选池；阈值由query标签事后定位，只是诊断。raw、实际部署路由和诊断曲线分开报告。

评分支线不更新任何模型，所有候选完整报告：margin_ref_rank_source1复用v7源valid预定分位网格、>=10changes/NAR<=1%/recover>damage选阈规则。extreme_q95/q99只由reference changed池>=32的margin_gain估计higher分位，必须gain>0。agreement_q95只由reference compatible池>=32拟合higher95分位；兼容条件为changed、gain>0、adapted head=adapted nearest source prototype、adapted candidate geometry margin>=frozen同候选margin。query仅应用固定阈值与逐样本兼容条件，禁止query归一化或fit。各policy同时报告sourcevalid/reference无标签采纳率/query最终风险，不能把极少改写包装成等纠错风险优势。不使用GMM或sigmoid(gain)。评分主候选仅为固定v6上的agreement_q95：两片Recovery均>=rank-source1减.5pp，且六cell每seed NAR<=1%才通过；低NAR本身不算成功。

预算：GPU0/1/2各一个source seed并行，不干预其他任务。最多60min/3GPUh、2GiB输出、每适应600s；只跑18项，不依据目标反馈增加超参/步数/切片。source-only预检通过后冻结执行；只允许有日志的实现修正。保存18checkpoint及完整query/valid预测、reference评分和聚类状态，独立重载全部模型/预测/聚类分配与策略，混淆矩阵复算、BN/原底座锁定和SHA审计。v7原封存不改，当前反馈记录在本run。
