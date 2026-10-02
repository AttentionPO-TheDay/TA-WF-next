# 多尺度TAM准确率方向并行筛查

用户授权同时验证前述提升方向，并允许CPU训练。当前多尺度生成器＋Transformer历史基线accuracy75.948%、Macro-F1 75.295%；上一深度宽度实验均未通过增量门槛，不重复加深/加宽。此轮把六个单因素方向及保序卷积分类器分开比较，不事先拼接改动，不开展漂移调整。24项新训练＋3项明确历史复用，具体候选如下。

## 数据、权限和表示

只用TemporalDrift原固定source102×150=15300和valid102×5=510。configs/datasets.json是只读原始路径入口，本轮不重新打开原始NPZ；prepared.pt字节复制上一run缓存，原行、标签、前5000原顺序有符号时间戳、方向条件2×1800 TAM保持不变。source标签只用于CE、获准source对比损失及拟合诊断；valid只用于20次预定选模和评价，不用于梯度、统计量拟合、适应或CPU/GPU调参。无未来日期、WTT/AWF、PCAP、预训练、teacher、TTA或query适应。

原绝对TAM按abs(float32时间戳)promote float64、floor(time*1799/80)、>=80归最后bin，正/负方向分通道。相对时间候选仅从相同5000包时间戳重建：每trace观测abs时间除以本trace最大abs时间，再乘1799、floor和clip[0,1799]，方向不变，时间戳0是padding不计数，空trace输出0。该归一化依赖自身X，无标签或valid拟合；不是证明同一物理时间差可信，不排序/差分/推断跨方向gap。输入总包数与各方向包数守恒，原source/valid标签和行相同。

除relative_time之外输入与原基线完全相同。span_mask只在source训练中施加，两方向同一连续90bin区间、每trace概率0.5、start均匀0..1710；source和valid评价无增强。它是缺失流量的训练扰动，不宣称真实网络漂移模拟。

## 候选与对照

|条件|设备|单项改变|对照|
|---|---|---|---|
|baseline|GPU0历史复用3seed|无新训练，75.948%|历史多尺度d2_w16|
|scale_concat|GPU0|3尺度输出拼接48通道，学习kernel1的48→16融合；分块I/3、bias0初始化|baseline|
|attention_readout|GPU0|128维零初始化query点积softmax读出，初始均匀；其余mean读出接口保留|baseline|
|relative_time|GPU0|上述逐trace相对时间TAM|baseline|
|span_mask|GPU0|上述source-only增强|baseline|
|supcon|GPU0|CE＋0.05监督对比损失，tau0.1|baseline|
|cosine_lr|GPU0|仅把三段LR替换为预定余弦日程|baseline|
|cpu_baseline|CPU|原基线从头训练，为设备数值对照|不作为新结构候选|
|cpu_temporal_cnn|CPU|仅把两层Transformer替换为两个保序时序卷积残差块|cpu_baseline|

固定多尺度生成器两层kernel5、16通道、GELU、共享dilation1/3/9，不复用训练权重。scale_concat保留同位置各尺度输出到新的线性融合阶段，未扩大token维度或给分类器额外输入；初始前向近似原等权平均，后续可学习跨尺度跨通道混合，同时增加784参数，不能宣称纯融合机制而无容量变化。attention query初始0，均匀权重；新增128参数且没有不可辨识bias。没有事后扫描mask强度、温度、loss权重或不同归一化版本。

其余固定：log1p计数；每token原30维计数保留，学习80维由16通道×5个有序3bin子区间均值，120×110时间token、全有效时间mask（空bin编码无流量）；100 packet槽位全mask且packet参数冻结。time projection110→128、位置和类型embedding，两层pre-norm Transformer d128/4heads/FF256/dropout0.1，最终LayerNorm与分视角读出256→102。模型从缓存TAM计算，relative_time的时间信息在预处理重建中使用，不增加packet分支。

cpu_temporal_cnn只处理120个时间token；两个块各为LayerNorm128、kernel3 Conv128→128、GELU/dropout0.1、kernel3 Conv128→128/dropout0.1及残差。同生成器、输入投影、时间位置、类型embedding、final norm和均值分类器，参数/感受范围/函数类不同，不称纯attention开关或算力匹配。CPU Transformer对照也从头训练；CPU卷积与CPU基线比较避免把设备数值差异归于结构。CPU−历史GPU baseline只报告描述差异。

supcon使用分类前时间读出128维L2归一化，同一batch同类样本为正例，排除自己；分母为所有其他样本，无正例anchor不计入均值，无正例整batch返回可微0。不改变抽样来增加正例，记录正例anchor比例；无valid标签、teacher预测或新分类头。CE和aux分别记录。

## 初始化、训练与选模

seeds21729/23407/22026；从头初始化，不加载历史训练checkpoint。公共模块初始state按同seed一致；尺度融合和query新增参数在隔离CPU RNG内构造；CPU卷积head不同，其余初始state一致。source randperm seed+9000+cycle按原规则拼成12800×64索引流，所有CPU/GPU任务与同seed历史baseline相同。增强用独立CPU generator seed+50000，不改变model dropout RNG或索引。

batch64、12800步、AdamW lr0.001/weight_decay0.0001，无AMP/TF32，确定性算法。baseline配方1–6400步lr0.001、6401–9600步0.0003、9601–12800步0.0001。cosine_lr仅改变为0.0001+0.00045*(1+cos(pi*(step−1)/12799))，从0.001到0.0001，不改optimizer或预算。

20次评价step3680+480*i，i=0..19；eval batch64，valid accuracy严格提高保存best、平分取最早、不early-stop。source同checkpoint记录拟合诊断，不参与选模。保存初始state/索引、首步梯度、每100步进度、完整曲线、best/latest含optimizer和RNG的checkpoint、best/last source/valid预测、best logits、参数变化、耗时/显存。CPU和GPU都重载best新实例核验预测，独立sklearn accuracy/F1复算。

## 执行前检查与历史复用

已完成7项CPU合成机制检查：共享初始化/初始恒等，packet不影响logits及batch独立/空输入有限，新增生成器/读出梯度，对比损失独立公式/单例类，相对时间计数与缩放不变性，连续双向同区间mask及独立RNG，LR端点。CPU/GPU各用source64样本前后向，无optimizer update/新增valid选模。真实source预检涵盖各候选有限梯度；同缓存相对TAM守恒及source/valid固定标签/行已核验。

历史baseline明确来自上一深度宽度run的d2_w16，其origin是先前多尺度run。新代码随机初始化与历史baseline初始state一致，GPU重载3份历史best并核验source/valid共6组完整预测一致，best/last指标在最终总管复算；历史3seed不是本轮新重复。历史预测/初始state/索引/曲线/报告为本run显式副本，原checkpoint仅用于重载审计，无训练warm start。源码来源见SOURCES.md，不隐式导入旧项目实现。

冻结config、PLAN、SOURCES、代码、两个cache、预检/准备报告、原datasets.json与历史引用hash至artifacts/freeze.json后才启动。训练入口拒绝draft并核验冻结hash，最终总管检查27报告×best/last×source/valid共108组指标、12800步/20次机会、最早最佳、全部初始化与索引一致性。

## 资源预算与停止

只用GPU0；实际为24564MiB RTX4090，GPU1/2已有高占用，不使用。最多3 GPU任务并发、每任务2CPU threads；最多2 CPU任务并发、每任务4threads（CPU队列同样包含source数据与完整训练，不只是审计）。三模型、三份完整source/valid GPU副本、前后向聚合预检峰值2.44GiB，额外计3GiB CUDA进程上下文、0.75GiB optimizer/cache裕量与4GiB保留，估计10.19GiB小于实际23.52GiB。该估计不是三独立worker运行实测，启动后仍核对实际显存。

18GPU＋6CPU新任务；GPU每任务3600秒、CPU每任务10800秒，包含训练与评价主循环；总管另容120秒启动/重载开销watchdog。整批36000秒上限，不增加任务/seed/步数。CPU真实source前后向Transformer约0.57秒、卷积约0.12秒，CPU完整对照可能数小时，GPU候选更快；诊断时间不承诺正式吞吐。没有全CPU Transformer重复其它六GPU方向，避免不必要的慢训练。

非有限值、冻结hash、权限、active generator梯度/参数变化、inactive参数、重载预测、初始化/选模/索引检查失败，或预算到期时，停止剩余任务并保留部分产物/错误；不得看到valid后自动重试参数、扩大预算、加模块或转用未来数据。实现错误修订须登记原因，冻后源码不静默改写。

## 预定判断与限制

主指标三seed平均valid accuracy；逐seed、Macro-F1、source拟合、best/last、耗时和参数同步报告。六GPU候选各对historical baseline，CPU卷积对新CPU baseline；候选门槛为均值accuracy增量≥1pp、3/3seed差>0、平均Macro-F1不降。相对通过与均值accuracy≥90%分别判断。CPU baseline只是设备控制，不作为不同方法创新或候选收益。满足门槛的候选才保留，若都失败仍保留历史多尺度开发基线；不自动拼接过门槛改动或继续调参。不同参数、输入、函数类、设备、耗时的差异不能唯一归因于一个抽象概念。

同一510 valid已多轮观察，七个候选并行增加选择偏差；门槛是开发一致性规则而非统计确认。三seed是优化重复，不是三个独立数据集。无源期结果能单独证明抗漂移或外部泛化；时间戳专属表示不能直接推广到WTT/AWF缺timestamp数据。漂移机制和外部评价仍须后续独立冻结。
