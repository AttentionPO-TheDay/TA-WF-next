# 20261001T113740Z_gpu_generator_view_ablation_35fd3e98

问题：固定当前可学习生成器+Transformer，packet-only、纯方向packet、TAM-only相对已完成融合模型各有何贡献？

状态：完成预检及历史融合重载审计后冻结配置，执行9项新训练；融合3seed明确历史复用，不重新训练。

用户授权本轮建议中的视角消融实验并行运行。目标为定位当前自有生成器+Transformer系统中packet、逐包时间幅值及TAM的增量，服务后续基础模型改进；不在本轮更换编码器、增加增强或启动漂移机制。整体源期valid三seed平均accuracy≥90%仍作为独立目标。

## 条件及解释

|条件|可见信息|新训练|
|---|---|---|
|packet_direction|packet方向、原顺序、有效长度；逐包时间幅值通道强制零；无TAM|3seed|
|packet_native|原packet视角：方向、原顺序、有效长度、80秒clip的时间幅值；无TAM|3seed|
|tam_only|原2×1800按方向分通道的TAM时间计数；无packet视角|3seed|
|fusion|原packet_native+TAM组合，原始220token可学习生成器+Transformer|复用3seed|

上一轮packet视角本身含时间，因此packet_native不得标为纯方向。TAM包含方向条件计数，也不能称不含方向的纯时间。增加packet_direction必要性：将packet逐包时间的贡献与加入TAM的贡献分开，避免把原packet视角移除误当时间模态移除。

本轮比较：fusion−packet_native识别加入TAM整体增量；fusion−tam_only识别加入原packet整体增量；packet_native−packet_direction识别当前packet编码中时间幅值的增量。另报告fusion−packet_direction、tam_only−packet_direction及单视角反向胜过fusion。它们都包括可用信息和参数参与自由度的差异，不能唯一归因于某个信号或证明不变量。

## 数据与标签权限

完全复用上轮`20261001T104959Z_gpu_multiview_generator_head_factorial_cf47d62c/artifacts/prepared.pt`，逐字节拷贝至本run并核验SHA256。路径来源为configs/datasets.json中proteus_temporal，数据审计沿用原manifest并注明reuse，无原始NPZ新访问、无重新划分。source102类×150=15300，valid102×5=510；每条前5000原存储顺序signed timestamp float32，以及经逐行重建核对的2×1800 TAM。零为suffix padding，不排序、不求跨方向包间间隔，不将timestamp称为包大小。

source标签仅CE训练和获准训练指标；valid标签只在20次预定机会选择checkpoint和计算指标，不进入梯度/适应。loader统一加载source/valid缓存，模型严格屏蔽未授权条件对应输入，CPU/GPU反事实测试核验被移除输入不影响输出。未来日期、WTT-Time、AWF、reference/query适应、预训练、teacher、TTA全部关闭。

## 结构和公平控制

显式复制上轮model.py为本run base_model.py，保留原局部CNN、固定统计、投影、位置/视角embedding、两层Transformer与分视角masked mean读出；未隐式导入旧项目。新model.py仅控制视角权限。每seed初始化与上一轮融合初始state逐tensor相等，state_dict名称和形状不变。

packet方向统计100个50包窗×5subbin，每subbin方向均值/有效比例/时间均值/方向切换比例，共20维；learned局部Conv1d3→16→16、kernel5、GELU，在原包序列聚合前编码，subbin均值展开80维，共100维。packet_direction时间均值与Conv输入时间通道均严格零；不是将时间置1后当作新测量。TAM log1p后120个15bin窗，固定2×15=30维加2层CNN后的80维，共110维。总槽位100 packet+120 TAM=220，d128，4heads，2layers，FF256，dropout0.1，最终两视角均值拼接256→102。

缺失视角在attention中全部屏蔽，读出对应半侧严格零，其独立CNN/投影/位置参数冻结且旁路。共享view_types与分类器仍是同一张量，未使用行/列没有有效梯度但可能受weight_decay影响；不因此宣称全部trainable参数都有效参与。synthetic空packet-only场景用恒定零key防止全mask NaN，该key不进入任何readout，不编码时间信息；真实source/valid无全空记录。TAM空bin为可见“无流量”，全部120窗有效。

|条件|总参数|可训练参数|生成器可训练参数|
|---|---:|---:|---:|
|packet_direction|350006|318966|1552|
|packet_native|350006|318966|1552|
|tam_only|350006|322726|1472|
|fusion|350006|350006|3024|

保持220槽位使分类器布局、dropout随机张量尺寸和共享初始化可比，但可见token数、有效容量、计算量不同。缺失分支不会以常量token参与attention。不得把此消融声称为参数容量完全匹配。

## 训练及选模

seeds21729/23407/22026；所有新训练从头初始化，不加载历史best模型。batch64，12800梯度步；source randperm按seed+9000+cycle拼接至819200条索引，与上轮相同seed融合完整相等。CE，AdamW lr0.001、weight_decay0.0001；1–6400步lr0.001，6401–9600步0.0003，9601–12800步0.0001；无增强。float32，无AMP/TF32，确定性算法，每任务2CPU threads。

20次source/valid评价steps3680+480*i（i=0..19）。仅valid accuracy严格提高时保存best，平分最早，无early-stop、valid梯度或valid驱动配方改变；评估batch64。Macro-F1同步报告。source指标为同一best checkpoint训练拟合诊断。

fusion reuse明确来源上轮learned_transformer同seeds，已完成12800步/20次选择及预测核验，mean accuracy74.183%。本轮显式加载其checkpoint仅用于重载审计，不用于初始化新训练；新fusion实现源码/初始state/真实source前向与旧模型精确一致，重载source/valid预测逐行相同。历史结果不是新增独立重复，也不代表融合又训练三次。历史cache及报告/预测hash冻结。

## 核验与裁决

CPU合成检查：初始化及原fusion输出一致；removed input反事实不变性；mask/readout隔离及全padding有限输出；active CNN梯度与inactive参数冻结。GPU64条真实source前后向，不执行optimizer，不评分valid；核验同样边界与双并行显存预算。历史fusion重载审计是既有source/valid成绩复现，不是未来性能开发。

训练保存初始state、完整采样索引、首步generator梯度、曲线、best/latest checkpoint、best/last预测、best logits、参数数目/变化、耗时。核验active token确实变化、inactive token零，inactive参数不更新。新实例重载新任务best预测并独立accuracy/F1复算；总管结束复算9新+3旧的48组best/last×source/valid指标，逐seed核验初始权重及抽样索引完全相等，保存逐seed错误转移（fusion改对/改错）。

主指标accuracy。开发增量门槛：三seed平均提升≥1pp、3/3seed差值>0、平均Macro-F1不下降；不是统计显著性确认。fusion相对packet_native及tam_only均通过才称本版有双视角增量证据。若任一单视角反向过同一门槛，优先审查融合路径；混合结果保持不确定，不将未过门槛当等价。所有条件90%达标与上述相对增量分开裁决。错误转移仅为现有valid事后诊断，不追加选模机会或回写训练。

## 资源与停止条件

GPU0最多2任务并发，不使用已占用GPU1/2。每新任务1小时上限、总管每任务watchdog额外60秒只作启动/结束开销；整批5小时上限。总量9×12800步，不追加seeds、超参、训练步数或条件。GPU预检两任务内存估计约6.22GB，GPU0可用约24.59GB。

非有限值、冻结hash/权限/梯度/重载失败或预算到期即停止其余任务，保留部分结果与错误记录；仅允许明确记录的实现错误修正，不隐蔽调参。method/config/code/data于训练前冻结在artifacts/freeze.json，入口拒绝draft。

本轮仍为固定已观察510 valid源期开发，不证明独立泛化或抗漂移。结果用于下一次有界编码/融合候选决策；漂移调整与外部缺时间戳适配需另行冻结权限和机制。本轮源代码复用与旧第三方环境来源见SOURCES.md。
