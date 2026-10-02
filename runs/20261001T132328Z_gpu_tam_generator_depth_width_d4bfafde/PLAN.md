# 20261001T132328Z_gpu_tam_generator_depth_width_d4bfafde

问题：固定多尺度及Transformer，TAM生成器2/4层×16/32中间通道能否改善源期准确率及泛化？

状态：CPU/GPU与历史基线审计通过后冻结；2×2生成器深度/中间宽度，三候选各3seed共9项新训练，原2层16通道多尺度3seed历史复用。

用户授权上一轮建议的下一轮实验。上一轮多尺度TAM生成器+Transformer源期valid mean accuracy75.948%，相对原局部+2.614pp、3/3seed正；尚未到90%，且训练拟合高不能证明增加容量一定有效。本轮固定已有效的尺度与分类器，只检验局部编码组合能力和容量，保留自有生成器→分类模型→后续漂移机制的接口。

## 固定数据与权限

沿用configs/datasets.json中的TemporalDrift路径来源及上轮prepared.pt，字节拷贝至本run核验相同SHA256，无原始NPZ新读取、重新抽样/划分或未来日期访问。source102类×150=15300、valid102×5=510，完全相同行/标签。缓存仍含前5000原顺序signed timestamp和审计一致的2×1800 TAM，但模型仅使用TAM，packet输入改变不影响logits，CPU/GPU反事实已核验。TAM方向条件计数+时间，不是不含方向的纯时间；log1p，80秒以上按原规则进入最后bin，不排序/求跨方向gap/伪造大小。

source标签CE训练及source诊断；valid标签只按20次预定机会选择checkpoint与评价，不进梯度、适应或参数规则。无未来日期/WTT-Time/AWF、reference/query适应、预训练、teacher、TTA或新增数据。固定510 valid已多轮开发，不重新称新确认集。

## 2×2深度与宽度设计

|条件|深度|中间通道|来源|生成器可训练参数|模型总参数|模型可训练参数|
|---|---:|---:|---|---:|---:|---:|
|A d2_w16|2|16|上轮multiscale_d139三seed复用|1472|350006|322726|
|B d2_w32|2|32|新训练3seed|2928|351462|324182|
|C d4_w16|4|16|新训练3seed|2016|350550|323270|
|D d4_w32|4|32|新训练3seed|5040|353574|326294|

各条件两层时间卷积kernel5，第一层2→width、最后一层width→16，固定dilation1/3/9分别作用于这两层，GELU。三尺度共享卷积权重，最终同一bin/channel做等权平均。新增深度在两层时间卷积之间加入两个kernel1逐点残差层z←z+GELU(Conv1d(z))，width→width，初始weight/bias均零。GELU'(0)=0.5保证非零branch梯度，恒等skip保留原初始编码和梯度路径；不新增norm、生成器dropout或scale gate。

深度4指两层kernel5加两层kernel1，总四个可学习卷积变换，不能称四层时间扩散卷积。零初始化pointwise残差不扩大时间支持，三尺度最大覆盖保持9/25/73bin（两时间卷积层理论跨度；dilation稀疏采样，不等于连续全部bin）。宽度16/32是中间特征宽度，最后均输出16通道，不追加token适配线性层。

所有条件固定：原TAM的每15bin×2方向完整log-count展开30维；学习特征按120token×5subbin×3bin有序均值，16×5=80维；120×110时间token，mask全有效（空bin是无流量）、100个packet槽位全masked，time投影110→128、位置/视角embedding、同两层pre-norm Transformer d128/4heads/FF256/dropout0.1、最终分视角masked mean及256→102分类器。packet独立参数冻结。共享view_types无效行/分类器无效半侧没有有效信号，可能有weight_decay但不引入packet信息。

同seed非generator初始state完全相同；同width的2/4层spatial卷积初始state相同，4层因新增branch零初始化，其源样本初始logits精确等于同width2层。额外模块在隔离CPU RNG下构建，恢复原随机流；新训练不加载历史best。宽度变化导致spatial权重形状/初始表示不同，是预定容量因素，不宣称完整generator初始化相同。

归因限制：depth比较同时改变残差组合、参数、非线性层数和计算，虽保持时间覆盖不变，仍不能唯一归因于抽象层数；width改变参数和函数自由度。所有条件输出/Transformer相同，但有效生成器容量不同。固定优化步数非固定GPU算力，均记录耗时/显存。较高source拟合不证明容量不足，本轮可能阴性。

## 训练、抽样与选择

seeds21729/23407/22026，从头初始化；source randperm seed+9000+cycle拼接至819200索引，同seed所有新条件与历史A一致。batch64、12800步，CE，AdamW lr0.001/weight_decay0.0001；1–6400步0.001、6401–9600步0.0003、9601–12800步0.0001；无增强。float32、无AMP/TF32、确定性算法、每任务2CPU threads。

20次source/valid评价steps3680+480*i（i=0..19），eval batch64；仅valid accuracy严格提高保存best，平分取最早，不early-stop，不按反馈改配方。Macro-F1同步报告，source仅同checkpoint拟合诊断。与历史A严格相同标签、数据、采样、步数、lr、选择机会。

## 历史复用与完成检查

原A来源上轮`20261001T124113Z_gpu_tam_multiscale_generator_101ab9e9/multiscale_d139`的相同三seed，mean75.948%。base_model.py/view_base.py/encoder_base.py显式本run代码拷贝并注明来源，无旧项目隐式sys.path依赖。audit_historical.py明确加载三个已完成checkpoint，只核验既有source/valid预测，不作为新条件初始化。A新代码与旧代码真实source前向精确一致，完整初始state相同，6组重载预测/12组best-last指标通过；旧结果/预测/索引标注历史复用，不作为新增独立重复。

5项CPU合成检查通过：共享classifier及同width spatial初始化；历史A和同width depth初始前向精确一致；固定token/interface/no-packet/batch独立/empty有限；在非零新增pointwise权重下time支持仍不扩大；所有active参数含零初始化残差都有非零有限梯度、单步可更新而frozen参数不变。GPU64条真实source前后向无optimizer更新/新valid选模，核验上述接口与双并行显存预算。

任务产物：初始state、索引流、首步generator梯度、完整曲线、best/latest checkpoint、best/last source/valid预测、best logits、generator参数/变化及耗时/显存。训练中active time token应更新，packet零；first step所有active generator参数梯度非零有限，best时全部active generator参数相对初始发生变化（特别新增残差），inactive参数严格不变。新实例重载各新任务best预测、独立accuracy/F1复算。总管结尾核验参数数/共享classifier/同width初始化/采样、复算9新+3旧共48组指标；错误转移只描述已观察valid，不使用真标签在线路由。

## 预定裁决

主指标accuracy；逐seed、Macro-F1、source拟合、best/last、参数量、耗时均报告。开发门槛：平均accuracy增量≥1pp、3/3seed差值>0、平均Macro-F1不下降，不视为统计确认。

分别报告宽度B−A与D−C、深度C−A与D−B、组合D−A；交互(D−B)−(C−A)报告数值，不只凭D胜A宣称协同。若任一新候选相对A过门槛，在合格者中按平均accuracy选优（平分生成器参数较少），保留为开发候选但不覆盖逐因素阴性；均未过时保留A，不自动继续扩大网络或追加学习率/尺度搜索。所有条件平均valid accuracy≥90%单独裁决，不用相对收益替代绝对性能。

## 资源与停止条件

仅GPU0最多2新任务并行，不使用GPU1/2现有占用。9项新任务×12800步，每任务1小时上限；总管watchdog额外60秒只用于启动/结束开销，整批5小时上限，不增加seed/步数/容量或修改尺度。GPU双任务预检显存估计约6.61GB，GPU0可用24.58GB，预检不承诺正式运行耗时。

训练前冻结config/code/data/history hash于artifacts/freeze.json，入口拒绝draft。非有限值、hash/权限/梯度/参数/预测复现失败或预算到期即停止剩余任务、保存部分结果和错误；不通过加预算改写失败。实现错误修订必须记录，不伪装为结果反馈改法。

本轮仍是已观察source/valid开发，三seed不是三个独立数据集。没有抗漂移评价/调整，达到90%也不等于独立泛化确认。依赖时间的版本不能直接用于缺timestamp的WTT/AWF；后续漂移机制及外部训练/评价权限另行冻结。
