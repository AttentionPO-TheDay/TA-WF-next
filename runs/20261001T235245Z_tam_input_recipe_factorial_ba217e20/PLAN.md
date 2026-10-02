# TAM幅值映射×优化配方2×2实验

用户授权下一轮实验。上轮GPU卷积替代73.203%失败，当前候选为自有多尺度生成器＋Transformer＋span_mask，valid mean accuracy77.647%/F1 76.889%。先完成RF差异审计（AUDIT.md），确认原计数、样本行和标签与RF历史缓存完全相同。RF85.948%仅历史参考，结构与配方等差异未定位单一瓶颈；本轮只比较输入幅值与优化bundle，不引入RF架构或继续加深加宽。

## 数据及权限

沿用configs/datasets.json路径来源与已审计prepared.pt字节副本。source102×150=15300、valid102×5=510，相同原始行、标签与5000包预算，TAM2×1800相同计数。无需raw NPZ新访问或重划分；RF native_prepared仅读取source/valid核对TAM、rows、labels逐元素一致。源时间戳0为padding，计数遵循80秒以上归最后bin、正负方向分通道的原规则；不排序/差分或制造包大小。模型仅TAM，packet槽位全mask，改原时间戳不能影响logits。

source标签CE训练/拟合诊断；valid标签仅20次固定checkpoint选择与评价，不做梯度/统计量拟合/适应。无未来日期、WTT/AWF、teacher、预训练、PCAP或TTA；旧checkpoint只baseline重载审计，新模型全部从头初始化。

## 条件

|条件|幅值映射|优化配方|来源|
|---|---|---|---|
|A log_current|log1p|原AdamW＋三段LR|原span_mask三个seed历史复用|
|B raw_current|原计数identity|原AdamW＋三段LR|3项新训练|
|C log_rfstyle|log1p|RF启发Adam/LR/wd bundle|3项新训练|
|D raw_rfstyle|原计数identity|RF启发Adam/LR/wd bundle|3项新训练|

raw/log1p同时用于固定30维计数与学习卷积输入，除此之外生成器/分类器/参数完全一样。原计数变换不新增信息权限，log1p也不是必然的信息损失；作用是幅值/优化适配假设，不预判结果。

所有条件保留两个kernel5 Conv2→16→16/GELU，共享dilation1/3/9，最终同位置等权平均；120token×固定30维原计数与16×5有序3bin子区间学习80维，共110维；100packet槽位全mask及其独立参数冻结，120时间token全有效（空bin为无流量）。同time projection110→128、时间位置/类型embedding、两层pre-norm Transformer d128/4head/FF256/dropout0.1、final norm、分视角均值与256→102分类器。350006总参数/322726可训练参数、生成器可训练1472，全部共享初始state逐元素相同；无新增norm/head/注意力读出或scale gate。

所有训练同source-only span_mask：每trace概率0.5，start均匀0..1710，双向同一连续90bin区间归零；每任务独立CPU generator seed+50000，保持同seed遮挡序列一致，不干扰dropout或抽样RNG。source/valid评价用完整输入，无增强。

## 训练与选择

seeds21729/23407/22026，batch64，12800步，source randperm seed+9000+cycle拼成相同索引流。A/B为AdamW lr0.001/wd0.0001、betas(.9,.999)/eps1e-8；步1–6400 lr0.001、6401–9600 lr0.0003、9601–12800 lr0.0001。C/D为Adam（coupled decay）lr0.0005/wd0.001、相同betas/eps，逐step lr0.0005*0.2**((step−1)/12800)。该日程按本轮固定步数归一化，不是native RF的batch200×30epochs按epoch日程；其变化是整体优化配方，不唯一归因于optimizer、wd、LR或日程。

CE，无AMP/TF32、确定性算法、每worker2CPU threads。20次source/valid评价step3680+480*i（i0..19），eval batch64；仅valid accuracy严格提高保存best、平分取最早，不early-stop。Macro-F1和同checkpoint source拟合同时记录，不返调规则。新任务保存初始state/索引、首步梯度、每100步进度、完整曲线、best/latest含optimizer/RNG、best/last source/valid预测、best logits及参数变化/耗时/显存。验证完整后新实例重载best复现预测并独立复算accuracy/F1。

## 审计、冻结及历史复用

已核对RF源码来源、原配方和输入、参数/结构、训练预算与机会，完整差异表见AUDIT.md。五项合成检查通过：初始化/历史log前向精确一致，输入固定特征映射/packet不影响/batch独立/空输入，有限非零generator梯度/优化器更新/frozen不变，Adam/AdamW与LR端点，mask RNG复现。GPU真实source64×3候选前后向有限、active梯度非零；无optimizer update或额外valid选模。三个历史A checkpoint以新代码重载完整source/valid，共6组预测精确一致；RF/current source/valid行、标签、原计数完全一致。

A原run span_mask三seed明确复用，不是新独立重复；复制预测/state/索引/曲线/报告，旧checkpoint不作为新训练初始化。B/C/D从头训练，CPU旧部分结果不参与本轮。代码本run显式副本，无旧项目sys.path依赖，第三方Python环境复用与来源见SOURCES.md。

训练前冻结PLAN/AUDIT/SOURCES/config/code/cache/预检/历史引用hash至artifacts/freeze.json，入口拒绝draft并验证hash。结束总管复核12报告×best/last×source/valid共48组指标、所有初始化/索引、步数/20机会/最早最大选模；raw输入没有参数新增。冻结文件不得根据结果静默修改。

## 资源、停止及预定裁决

只GPU0 RTX4090 24GB，最多3worker并行，GPU1/2已占用不使用。不再CPU慢训练，CPU只审计与调度。三模型聚合前后向＋三份数据＋进程上下文/optimizer缓存/4GiB安全余量估计10.17GiB低于实际23.52GiB，启动后核对实测。

9项新任务各12800步；每项3600秒、总管另120秒启动/重载watchdog，整批18000秒上限。不增加seed/步数/输入映射/优化器或增强强度扫描。非有限值、hash/权限/梯度/参数/重载预测/选模检查错误或超时停止剩余任务，保留部分结果；实现错误修订必须登记，不靠结果返调隐藏修复。

主指标accuracy，均值增量≥1pp、3/3seed正、平均Macro-F1不降为开发门槛，≥90%单独判断。分别报告B−A、C−A、D−B、D−C、D−A；交互(D−B)−(C−A)报告数值而不声称统计确认。新候选对A通过才保留，在通过者中按mean accuracy选优；全失败则保留A，不自动合并未过因子或扫描参数。逐seed、source拟合、best/last、参数/耗时及阴性结果全保留。

固定510 valid已多轮观察，三seed是优化重复，2×2筛查不是新确认；源期改进不等于抗漂移或外部泛化。RF同数据原native性能是不同结构/配方的参考，不能因本轮正向宣称定位RF优势唯一原因。本轮未来、适应与外部评价仍关闭。
