# 生成器逐bin通道归一化×逐级时间编码

用户授权下一轮实验。上轮raw输入和RF启发优化bundle均失败，保留log1p＋原AdamW三段LR＋多尺度生成器＋Transformer＋span_mask77.647%为开发对照。本轮仅改变生成器，比较局部通道LayerNorm与两层之间的3bin聚合，不重复token后的卷积分类器替代、不加深加宽、不改input/LR/增强强度。

## 数据和权限

configs/datasets.json为数据来源；prepared.pt字节复制前run缓存，原source102×150=15300、valid102×5=510、行/标签/TAM和前5000包预算不变。时间戳0是padding，方向条件TAM2×1800仍按80秒及以上归最后bin规则，所有候选log1p。缓存时间戳保留但packet全mask，改变timestamp不影响logits；不排序/求跨方向差分/重划分，不新读raw或未来数据。

source标签只用于CE及source拟合诊断，valid只20次固定checkpoint选择和指标，不参与梯度或归一化参数统计拟合。LayerNorm是每样本每bin通道内统计，不用batch或时间整体统计，无running stats/目标集适应。无未来日期、WTT/AWF、teacher、预训练、PCAP或TTA。新模型从头初始化，历史checkpoint只baseline复现审计。

## 2×2设计与精确结构

|条件|通道归一化|时间编码|来源|
|---|---|---|---|
|A flat_none|无|两层在1800bin上编码，末端3bin均值|3seed历史mask基线复用|
|B flat_norm|两层各LayerNorm16|同A|3新训练|
|C hier_none|无|第一层1800bin、3bin均值→600、第二层600bin|3新训练|
|D hier_norm|两层各LayerNorm16|同C|3新训练|

所有候选仍两个kernel5 Conv2→16→16、GELU，分别独立在dilation1/3/9上编码，同卷积权重及可选归一化参数跨尺度共享，三尺度同位置等权均值。归一化在每次conv之后/GELU之前，输入转置BxTx16后LayerNorm16 eps1e-5，gamma1/beta0初始化，两个独立norm共64可训练参数。它改变表示与优化自由度，不称参数严格相同。

flat两层都1800bin，最终3bin均值成600个学习子区间；hier第一层后保序3bin均值成600bin，再在600网格第二层卷积，不再末端3bin均值。全流程不改变conv层数/通道，层级候选改变聚合位置、非线性组合顺序、采样网格和时间支持，是完整编码设计因素，不唯一归因为抽象层级。

支持跨度：flat第二层单bin的最大原时间支持9/25/73bin，最终3bin均值学习子区间11/27/75bin；hier第二层600网格输出学习子区间19/51/147bin。dilated支持可能稀疏，不等于跨度中所有bin都使用。合成输入梯度核验最终600网格对原输入支持端点及3bin网格对齐，避免将不同网格数字误作同一位置比较。

固定计数token始终从原1800 log-count按2方向×15bin完整展开30维，未随层级或norm修改。600学习子区间按120token×5有序子区间×16通道展开80维，时间token120×110、全有效mask（空bin表示无流量）、100packet槽位全mask及其参数冻结。即使学习分支聚合方式改变，固定原计数仍保留；不能声称此实验消除或引入全输入信息损失。

Transformer、投影、位置/类型embedding、读出完全固定：d128/4heads/2层pre-norm/FF256/dropout0.1，final norm、分视角masked mean＋linear256→102。无卷积分类器替代、BN或新scale gate。

## 初始化、训练与选择

seeds21729/23407/22026；公共state（全部旧generator卷积与Transformer）同seed逐元素相同；norm在隔离CPU RNG构造，gamma1/beta0，不改变公共初始化。A新代码与历史log_current初始及完整best预测精确一致。B/C/D因norm/网格改变初始前向不同，不能称四条件初始logits相同。

source randperm seed+9000+cycle拼成12800×64索引，所有条件与历史A同seed一致。batch64、12800步、CE、AdamW lr0.001/wd0.0001、betas(.9,.999)/eps1e-8；steps1–6400 lr0.001、6401–9600 lr0.0003、9601–12800 lr0.0001。source-only span_mask固定每trace p0.5、双向同一连续90bin、start均匀0..1710，独立CPU RNG seed+50000，同seed候选相同遮挡序列、不影响dropout/抽样；source/valid评价无增强。

无AMP/TF32、确定性算法、每任务2CPU threads。20次评价steps3680+480*i、i0..19，eval batch64；valid accuracy严格提高选best、平分最早、不early-stop，source同checkpoint仅拟合诊断。保存初始化/索引、首步梯度、每100步进度、完整曲线、best/latest含optimizer和RNG、best/last预测、best logits、参数变化/耗时/显存；新实例重载best预测和独立sklearn accuracy/F1核验。

## 复用与执行检查

6项合成测试通过：公共初始化与baseline精确前向、固定token接口/原计数/batch独立/packet无影响/空输入有限，所有active generator参数含norm有效梯度/可更新与frozen不变，norm不跨batch/time且无running stats，最终学习子区间支持与网格对齐，mask及原优化器复现。GPU真实source64×3候选前后向及norm梯度检查通过、无optimizer update；3历史baseline checkpoint完整source/valid共6组重载预测精确一致，旧结果仅复用非新重复。

代码/输入为本run显式副本，来源SOURCES.md，无旧项目sys.path/实现依赖。训练前冻结PLAN/SOURCES/config/code/cache/预检/历史引用hash至artifacts/freeze.json。入口拒绝draft、核验hash。末尾总管12报告best/last×source/valid共48组指标、12800步/20机会/最早最大accuracy、索引/公共初始化、norm初值及参数数量复核。所有active generator参数best相对初始更新，inactive严格不变。

## 资源与停止

只GPU0 RTX4090 24GB，GPU1/2已占用不使用，最多3worker并行，不再CPU慢训练。三模型聚合前后向、三份数据、上下文/optimizer缓存及4GiB安全余量估计10.19GiB小于实际23.52GiB，启动后查看实测。9项新任务各12800步、每任务3600秒、总管另120秒启动/重载watchdog，整批18000秒上限，不新增seed/步数/Norm种类/聚合倍率或mask搜索。

非有限值、hash/权限/梯度/参数/重载/选模/初始化检查失败或超时停止余下任务、保存部分结果，不通过追加预算/未来访问绕过。实现错误修订登记，冻后文件不按结果静默修改。

## 裁决与限制

主指标accuracy，增量平均≥1pp、3/3seed正、平均Macro-F1不降为开发门槛，平均accuracy≥90%单独判断。报告B−A归一化、C−A逐级编码、D−B逐级编码、D−C归一化、D−A组合和交互(D−B)−(C−A)，交互仅数值描述。对A过门槛的新候选才保留，通过者mean accuracy最高优先，全失败保留A77.647%，不自动拼接未通过因素或返调参数。

固定510 valid多轮观察，三seed是优化重复而非独立数据集；2×2开发筛查不是新确认。两个因素的容量/网格/非线性/感受范围变化不证明单一抽象原因；源期收益不证明外部泛化或抗漂移，也不保证达到90%。未来数据和适应仍关闭。
