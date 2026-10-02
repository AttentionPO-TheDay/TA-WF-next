# 20261001T124113Z_gpu_tam_multiscale_generator_101ab9e9

问题：同TAM输入、token接口及参数量下，扩展局部上下文和共享权重多尺度编码能否改善生成器+Transformer？

状态：预检及历史TAM-only重载通过后冻结；2候选×3seed共6项新训练，原局部编码3seed历史复用，不重复训练。

用户授权下一轮建议的TAM单视角生成器对照。当前自有生成器+Transformer的融合源期accuracy74.183%，TAM-only73.333%，fusion−TAM +0.850pp且2/3seed正，未过增量门槛。本轮只验证扩大局部上下文/组合多个尺度的编码设计；不认定生成器编码已是唯一瓶颈。保留自有生成器→分类模型→后续单独设计漂移调整的研究接口。

## 数据与信息权限

沿用configs/datasets.json的TemporalDrift source102类×150=15300、valid102×5=510，直接字节复制上轮artifacts/prepared.pt至本run并核验原hash。manifest.json标明上轮来源；不重新划分、不新读raw NPZ。TAM是原存储顺序前5000有符号时间戳构建的2×1800计数，abs(timestamp)≥80秒进入最后bin，方向分通道，log1p。逐行重建一致性与完整valid隔离审计沿用上轮冻结缓存，不伪造时间戳/包大小、不排序或计算跨方向间隔。

三条件均只给模型TAM，原packet视角完全屏蔽，raw signed timestamp幅值/方向变化不能影响logits（CPU/GPU反事实检查）。TAM本身包含方向条件的时间计数，并非不含方向的纯时间模态。source标签仅用于CE及训练指标；valid仅预定checkpoint选择和获准评价，无valid梯度、适应或搜索。未来日期、WTT-Time、AWF、reference/query适应、预训练、teacher/TTA全部关闭。

## 参数匹配的三个编码

|编码|两层卷积dilation|上下文跨度bin|来源|
|---|---|---:|---|
|local_d1|1/1|9|上轮tam_only三seed复用|
|context_d9|9/9|73|新训练3seed|
|multiscale_d139|同时1/1、3/3、9/9，共享两层权重|9/25/73|新训练3seed|

每尺度均在完整1800bin上用Conv1d(2→16→16,kernel5,padding=2*dilation)+GELU。多尺度在相同bin/channel对三套GELU后结果等权算术平均；不跨时间位置/方向打乱，也无新增scale参数或生成器dropout。权重共享为同一组4个参数张量，不复制三组卷积。context_d9与多尺度的最大理论上下文同为73bin，用以避免把更大最大感受野误当多尺度独有收益。

按标准80/1799秒间隔，9/25/73bin的首末跨度约0.356/1.067/3.202秒；跨padding与最终80秒聚集bin时不能当均匀物理时间跨度。此为两卷积层的理论覆盖跨度，dilation3/9只有对应间距的稀疏支持，不能称连续看到25/73个全部bin；subbin均值进一步增加输出token的覆盖。本轮只测试预定d=1/3/9，结果后不追调尺度。

所有条件保留固定30维原log-count：2通道×每token15bin按原顺序展开；learned通道在每15bin token的5个3bin subbin平均后展开16×5=80维。120个时间token×110维、每token的计数顺序、有效mask、投影及读出相同。原local已有保序统计，本轮并非把原来无序表示改成有序。

保留原模型布局100个packet槽位全部masked、120个TAM窗均可见（空bin=无流量，非padding）。packet独立CNN/投影/位置参数冻结；共享classifier无效半侧/view_types无效行仍在张量中，不是有效新增信号。分类器为同两层pre-norm Transformer，d128、4heads、FF256、dropout0.1，最终LayerNorm、per-view masked mean、256→102读出。不会同时换分类头或融合路径。

三条件总参数350006、trainable322726，generator总3024/可训练1472完全一致；同seed完整初始state逐tensor相同，包括分类器。构造CNN模块后仅通过F.conv1d改变dilation，未消耗额外初始化随机流。整个encoder命名参数仍相同，所以新任务与历史local完整权重/采样可核验。

解释边界：多尺度两层局部卷积调用约为单尺度三倍，固定优化步数不代表固定GPU计算；参数数相同不代表函数类相同。多尺度与d9差值同时包括邻域密度、不同跨度组合与计算量，平均也改变初始特征分布，不能凭正向结果唯一归因于抽象“多尺度”。稀疏卷积可能产生栅格效应。保序不意味着可逆或抗时间漂移不变量。

## 训练与选模

seeds21729/23407/22026，从头初始化，不用历史best warm start。source按seed+9000+cycle的randperm拼接，batch64、819200索引=12800步；与相同seed历史local逐项相等。CE，AdamW lr0.001、weight_decay0.0001；1–6400步0.001，6401–9600步0.0003，9601–12800步0.0001，无增强。float32、无AMP/TF32、确定性算法、每任务2CPU线程。

20次source/valid评价在steps3680+480*i（i=0..19），仅按valid accuracy严格改进保存best，平分选最早，无early-stop，无按反馈改学习率/步数/结构。eval batch64，Macro-F1同步报告；source仅同checkpoint拟合诊断。与历史local保持相同训练标签、采样、梯度步数、学习率及选择机会。

## 历史复用与检查

上轮`20261001T113740Z_gpu_generator_view_ablation_35fd3e98`的tam_only三seed已完成，均值73.333%。base_model.py及view_base.py显式逐字节复制至本run，后者来源为上轮model.py，不隐式导入旧项目。local_d1 GPU源码路径与原模型前向精确一致，新seed与历史初始state相同，audit_historical.py显式重载原checkpoint并核验原source/valid共6组预测、12组best/last指标。checkpoint仅用于旧成绩复现，不进入新训练。旧结果、预测、初始化与索引复制进本run并标注historical_reuse，不算新增独立重复。

5项CPU合成检查通过：相同参数/初始化与原编码精确一致；尺度等权共享权重的独立重建及保序统计；dilation真实支持范围；packet不泄漏/empty有限/batch独立；active CNN非零有限梯度与冻结参数不变。GPU64条真实source三条件前后向无optimizer更新，核验同边界与并行显存预算，无新valid性能选择。

任务保存初始化、索引流、首步generator梯度、完整曲线、best/latest模型、best/last source/valid预测、best logits、参数变化、耗时/显存。active time token应变化，packet始终零且inactive参数不更新。新实例重载新任务best逐行复现并独立accuracy/F1计算。总管结束核验参数数、全部初始权重及索引，复算6新+3旧×best/last×source/valid共36组指标；错误转移只作已观察valid描述，不用于query真标签路由。

## 预定裁决

主指标accuracy；同步逐seed、Macro-F1、source拟合、best/last、计算量。开发候选门槛为三seed平均accuracy提升≥1pp、3/3seed差值>0、平均Macro-F1不下降，不是统计显著性确认。

分别比较context_d9−local_d1、multiscale_d139−local_d1、multiscale_d139−context_d9。multi同时胜local与context才支持本设计超出只扩大最大跨度的价值；若仅context胜local而multi未胜context，优先保留较低计算上下文候选；multi只胜local时不能确认多尺度特异收益；未过则保留阴性/混合证据，不自动追加尺度或正则化搜索。所有条件平均valid accuracy≥90%目标单独裁决，不用相对门槛代替。

## 预算与停止条件

仅GPU0，最多2新任务并行，不使用已占用GPU1/2。共6×12800步，每任务1小时上限；总管watchdog额外60秒仅作进程启动/结束开销；整批4小时上限。不追加seed、尺度、步数或修改输入规则。GPU双任务预估显存约6.17GB，GPU0可用约24.58GB。多尺度单任务耗时需记录，预检不是正式完成时间承诺。

冻结前检查代码语法、CPU/GPU及历史审计、数据hash；机器配置见config.json，冻结hash见artifacts/freeze.json，训练入口拒绝draft。出现非有限值、hash/权限/梯度/复现异常或预算到期即停止其余任务、保留产物与日志，不通过加预算修饰失败。实现错误修正须单列说明，不将结果反馈改法伪装修错。

本轮固定510 valid已多次开发，不是新确认集，三seed不是三个独立验证集。即使源期性能改善或达到90%，尚未证明抗漂移；缺时间戳的数据集不能自动使用此版本。下一阶段漂移机制/训练权限/评价对照须另行冻结；本轮不执行。
