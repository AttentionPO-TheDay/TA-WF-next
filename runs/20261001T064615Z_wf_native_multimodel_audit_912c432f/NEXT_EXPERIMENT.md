# 下一轮多模型比较草案（未启动）

目标：固定source valid510、102类、seed21729/23407/22026平均accuracy>=90%，Macro-F1及逐seed同时报告，不用最佳seed代替均值。全部训练仍source150固定15300条，不以扩数据/换验证集掩盖比较；目标只是源期，不开放未来。

## 优先队列

1. Var-CNN官方支持dir-only配置：按固定源代码最小显式移植、保留MIT声明，先做结构及因果padding/SAME pool合成验证。batch50、Adam.001、至多150epoch、plateau patience5/factor sqrt(.1)/minlr1e-5、accuracy earlystop patience10；在新run冻结Keras2.0.8隐式eps、回调min_delta及BN约定后执行。三个seed统一使用21729/23407/22026（此处覆盖分报告中的建议seed1729/3407/2026）。每epoch看valid是native规则，最多150次，不与历史20次机会混称匹配。全source不新增5%随机拆分、关闭metadata/time；输出称“官方dir-only结构和配方、本项目固定split适配”。
2. RF作者模型+TAM：仅当源期timestamp/TAM接口审计通过后训练。102类，5000包→2×1800，80秒窗口，30epoch/batch200/Adam5e-4/wd.001，lr=.0005*.2**(epoch/30)。20预定选模epoch，accuracy earliest max，报告last以对齐作者最终模型。相较方向模型增加时间信息，独立列为原生输入性能，不作网络单因素结论。
3. 复用DF70.784% accuracy与当前CNN73.987% accuracy作历史参照，不自动追加训练。DF60轮候选只列后备，不同时启动。ARES暂缓到输入长度及随机eval问题解决，不能标成已有ready候选。

## 公平解释与选择

本批首先比较各自原生训练流程的实际性能；训练步数、优化器、参数及验证机会不同，全部记录，不宣称容量/优化/选模预算匹配。Var-CNN若在该流程下明显更高，只能先说明完整流程优于历史流程。需要归因时另立同输入、同数据、同验证机会和预算的第二阶段，不能在看到结果后隐去其更大调参预算。

新的正式run统一以accuracy为选模主指标、同值取最早；历史以F1选模事实保持不变。DF已有20点曲线两种规则选中同一checkpoint已审计；CNN历史数值保留原F1规则，不能擅称按accuracy挑选。达到90%需明示是哪个模型/输入/选择规则，非原生成器自动达到90%。不按外部未来反馈调参。

预算草案：GPU0最多2任务且先做两模型并发显存试验；Var-CNN最多1实例同时。每seed硬上限60分钟；Var-CNN3seed最多3GPU小时，RF3seed最多3GPU小时；总批墙钟上限6小时、不自动扩预算或换LR重试。预算触顶保留部分结果，不称充分收敛。训练器必须拒绝draft；本文件没有启动任务。

检查门槛：版本/hash冻结、source150/valid行与标签对齐、fullvalid重复隔离沿用审核清单、初始随机模型与输入有限、gradient/CUDA显存、选模与未来权限锁定、逐条pred及独立指标复算。源期时间审计只读取必要train/valid，不读未来TAM、旧checkpoint或非本项目训练包。
