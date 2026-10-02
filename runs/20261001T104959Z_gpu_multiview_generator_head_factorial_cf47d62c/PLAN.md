# 20261001T104959Z_gpu_multiview_generator_head_factorial_cf47d62c

问题：相同方向+时间信息下，固定/可学习多视角token生成器×MLP/Transformer是否改善基础模型，并据结果评估整体框架？

状态：配置冻结后执行12个GPU训练任务；仅源期开发，未来评价关闭。

用户目标：先建立自己设计的生成器+分类模型，目标为固定源期valid三seed平均accuracy≥90%；再为该系统设计缓解流量时间漂移的调整机制。本轮只回答基础框架与可学习局部特征的增量，不将源期准确率提高称为抗漂移。

## 数据与权限

TemporalDrift固定source每类150条、102类共15300条，valid每类5条共510条。完全沿用已观察选择清单，不重划分；来源由configs/datasets.json定位。每条只使用原存储顺序前5000有符号时间戳float32及同一记录的2×1800 TAM。零为suffix padding，不排序、不计算跨方向包间间隔、不伪造包大小。TAM逐行独立重建与上一轮原生RF输入一致，细节与hash见manifest.json、DATA_AUDIT.md。

source标签仅用于CE梯度和获准训练指标；valid输入与标签仅用于预定checkpoint选择及评价，不用于梯度或适应。无未来日期、WTT-Time、AWF、参考库、query适应、预训练或外部checkpoint。训练从头初始化。数据准备已读取train/valid的完整NPZ成员并只分析固定选择行，此访问记录不算新确认数据。

## 模型与四个条件

|条件|生成器|分类器|
|---|---|---|
|A fixed_mlp|固定局部统计|逐token残差MLP|
|B fixed_transformer|固定局部统计|Transformer|
|C learned_mlp|固定局部统计+可学习CNN局部特征|逐token残差MLP|
|D learned_transformer|固定局部统计+可学习CNN局部特征|Transformer|

packet视角5000包按100×50分窗，每窗5个10包subbin。每subbin统计方向均值、观测比例、clamp(abs(timestamp),80)/80的均值、相邻有效包方向切换比例，共20维。learned条件在原5000包方向/观测/归一化时间三通道上用两层Conv1d(16通道,kernel5,padding2)+GELU，每层显式padding屏蔽，然后每subbin做masked mean，得到80维追加特征；fixed追加80维零，packet token共100维。

时间视角对TAM做log1p，按120个15bin窗口切分，两通道原计数展开成30维固定特征。learned条件在两通道完整1800bin上用两层相同规格CNN，按每窗口5个3bin subbin均值展开80维；fixed追加80维零。每time token110维。共100 packet+120time=220token，两条件mask相同。packet全空窗不可见；所有time窗可见，空bin表示无流量而非padding。

固定CNN存在但被冻结且旁路，仅保证共享模块初始化顺序。可学习CNN共3024参数，梯度来自source CE；特征投影一直属于分类器，不改名为生成器贡献。两个视角各自Linear→128，加可训练位置/视角嵌入。MLP两层pre-norm残差逐token128→512→128；Transformer两层d128、4heads、FF256、pre-norm。dropout均0.1。最终LayerNorm、两视角独立masked mean、拼接256→102分类。

|条件|总参数|可训练参数|生成器可训练参数|
|---|---:|---:|---:|
|A|348982|345958|0|
|B|350006|346982|0|
|C|348982|348982|3024|
|D|350006|350006|3024|

限制：fixed额外通道为零，learned填入CNN特征，有效表示和分类器投影参与自由度不同。C−A、D−B衡量本编码设计整体增量，不能唯一归因于参数学习。MLP/Transformer名义参数接近亦不代表关系建模能力完全匹配。本实现是自有系统的新版本，未直接迁入DF/RF生成器，也不凭独立编程认定研究原创性。

## 训练、选模与审计

训练seeds=21729/23407/22026，每seed四条件共享非head初始化；同head的fixed/learned完整初始state相同。每seed所有条件使用相同15300条source按seed+9000+cycle生成randperm拼接的819200条索引序列。batch64、12800步，约53.54个source遍历。无数据增强，用以避免只遮方向而保留TAM造成多视角输入不一致；不沿用此前纯方向mask配方。

AdamW lr0.001、weight_decay0.0001；1–6400步lr0.001，6401–9600步0.0003，9601–12800步0.0001。float32，无AMP/TF32，确定性算法，2CPU threads/task。20次valid评价，steps=3680+480*i,i=0..19，最佳valid accuracy严格提高时保存，平分选最早；不早停、不按valid改超参。source指标只作同checkpoint诊断。

先完成5项CPU合成检查及64条真实source GPU前后向预检，预检不执行optimizer、不评分valid。共享权重、mask、两分支非零有限梯度、fixed无梯度和显存预算通过。训练中检查固定token不变、学习token两视角变化；保存完整曲线、配置、初始化、索引、best/latest checkpoint、source/valid best及last预测、best logits。任务结束新实例重载best并复现预测、独立计算accuracy/F1；总管再次复算全部48组best/last×source/valid指标及初始化/抽样一致性。

## 冻结裁决

主指标accuracy；同时报告Macro-F1、逐seed、source拟合、best与last和耗时。预定增量门槛为三seed平均accuracy≥1pp、3/3seed差值>0、平均Macro-F1不下降，作为开发候选筛查，不声称统计确认。

分别报告生成器C−A、D−B；Transformer B−A、D−C；目标组合D−A。交互(D−C)−(B−A)报告逐seed与均值，不仅凭D胜A称协同。绝对90%目标单独判断每条件平均valid accuracy≥0.90，不用相对增量替代。

框架决定规则：若D侧生成器和Transformer增量均过门槛且D最高，保留生成器+Transformer候选；若C侧生成器过门槛但D−C未过，优先生成器+MLP候选；若两个生成器增量均未过，重审信息保留/汇聚设计，不直接叠加漂移模块；混合证据不强行作唯一决定。未过门槛不代表某架构普遍无效。详见FRAMEWORK_REVIEW.md。历史RF accuracy85.948%、方向CNN+MLP+mask73.987%只作已观察性能参照，输入表示/训练配方/选模机会不同，不作严格机制对照，不加载其checkpoint。

## 预算与边界

仅GPU0，最多2任务并行；12任务、每任务1小时上限（启动总管watchdog额外60秒仅用于结束开销），整批6小时上限。GPU1/2已有任务，不使用。预检双任务显存估计约6.23GB，GPU0约24.59GB可用。全部异常、非有限值、冻结hash失配、梯度/重载/数据边界失败即停止余下任务并保留结果；不得通过追加预算或修改方法继续同run。

预检计时只是无更新单任务前后向，不能承诺实际完成时间。机器配置见config.json，代码/配置/数据冻结hash见artifacts/freeze.json。只复用旧第三方Python/PyTorch环境与本项目已审计数据，没有sys.path旧代码或旧checkpoint。

第二阶段需另立实验：先冻结基础系统，再审计TemporalDrift日期衰减、定义调整信号、更新参数范围和reference/query及标签权限，并对比原系统静态、不调整、相同预算通用调整。当前时间输入版本需明确缺时间戳的跨数据集适配方案，不能自动直接推广到WTT/AWF。当前510 valid多次参与开发，即便达到90%也不证明独立泛化或缓解漂移。
