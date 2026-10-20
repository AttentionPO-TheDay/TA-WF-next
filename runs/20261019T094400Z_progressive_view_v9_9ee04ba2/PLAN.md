# 渐进平滑梯度平衡与独立输入辅助视图 v9

用户授权保留优化梯度平衡并探索表征解耦；本轮停止完全去KL及更高分位拒绝分支，不重复它们。本轮是v8反馈影响的已观察Version045/USA两片诊断开发，不是新确认。沿用18,553纯source/102类、三个seed42/100/2026、不可变1024无标签reference与独立query；source BN参数/统计与所有原模型完全锁定，仅Q/V r4 LoRA4096参数更新。完整Teacher KL、.1pull+relation、source identity1、原记忆及软目标保持。没有源backbone或额外神经分类头训练，没有WTT/AWF新评分。

v8已是动态EMA梯度比值，35倍只是均值；本轮不把它写成固定倍率。匹配历史控制为v8 geometry_grad_balance和v6 sharp_target_only，SHA核验而不重训。新增四arm×两片×三seed=24适应：ramp_balance仅在v8倍率上加100步sin²渐进；smooth_balance在该相同渐进下，再将目标log倍率EMA.9，初始0，每步log变化限.02，200步后限.005；smooth_view再添加固定源辅助视图蒸馏；smooth_shuffled_view对应关系打乱对照。所有400步、Adam .00025、batch64、source identity32、原初始化/批流20261019、clip1、固定final400，无目标选模。独立测量倍率平滑和梯度/更新方向，倍率平滑不等于方向必稳。

辅助视图从原始2×1800 TAM在1/6/24/120时间块聚合，得到302 log1p方向计数；仅18553source拟合列标准化、PCA32正交投影再白化，以及102类中心。它读取与主模型共享的原始流量，但其距离没有使用主128维编码器，所以不是同一几何空间的一次等距旋转。仅source-valid2160标签从固定[.03,.1,.3,1,3,10,30]选择prototype距离softmax温度最小NLL；sourcevalidF1>=.1且NLL<log102才启用辅助，否则所有arm照跑、辅助记为关闭。source-view拟合一次三模型共同使用，不引入新源神经训练。源source/valid行与访问权限显式限定，目标reference只做前向；query绝不fit任何投影、原型、温度或CDF。

辅助损失=.25 mean(w*c*KL(student_logits/T2,固定raw-view q)*T²)，c=1-H(q)/log102；只在适应时提供另一个表示的证据，主推理仍原分类头+LoRA，无query后验融合。打乱对照每batch对(q,c)一起permute，再按当前w重标定保持自身该batch加权总质量，与真实视图相比检验样本对应信息，而不把效果归给多加一个loss。跨臂状态演化仍会使权重不同。必须通过原模型错误集合内的view纠错/同错等指标实测互补性；PCA正交、辅助头或更多视图均不自动证明独立语义。

预注册判据详见config.v9.main_comparisons。主稳定性比较smooth/ramp，两片后200步倍率log变化量平均降>=30%、总clip不增加，F1及同margin Rec@NAR1各下降不超过.1pp。单独效果判据smooth/v8两片F1不降、NAR增<=.2pp、Rec@NAR1各增>=.2pp。辅助证据必须真实视图相对打乱两片共错纠错各>=+.5pp、F1不降/NAR增<=.2pp，并且相对无辅助smooth，F1与Rec@NAR1不降。所有指标和次候选保留，不能挑最优臂替换预注册比较。

评价与v8同口径：逐seed F1/Accuracy/NAR/Recovery、无sigmoid raw margin gain的query诊断曲线（标签定位阈值不能部署）、仅reference KMeans102拟合与query purity/ARI/NMI、固定frozen head=source prototype=双bank同错subset的still_same_wrong/corrected/other_wrong。另报告独立输入视图单独sourcevalid/query表现、对frozen错误和共错集合的改对率与同错率。没有新增风险拒绝/极端分位/GMM策略，raw主输出不加安全门控。

预算GPU0/1/2各seed、24适应仅一次；max60min/3GPUh/2GiB、每cell600s。source-only预检、所有24checkpoint及历史控制完整reference/query/valid重载、独立混淆指标/聚类分配、梯度倍率与总质量审计、输入/输出SHA。数据原始只读；失败仅修实现错误并记录，不因目标反馈追加arm或改变参数；不扩展全矩阵。v7/v8封存不改。
