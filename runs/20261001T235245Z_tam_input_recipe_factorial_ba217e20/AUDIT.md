# RF与当前自有模型的源期差异审计

已核对本项目明确迁入的RF源码/worker/config与自有TAM生成器/Transformer/worker。RF来源为https://github.com/robust-fingerprinting/RF ，commit b75680148429c5f554114b78a0f846541b39f0ec，文件RF/models/RF.py的sha256见前原生基线run/rf_native.py；本轮不引入RF架构作为自有生成器或声称其原创。

|维度|RF原生历史实验|当前自有mask候选|
|---|---|---|
|样本与标签|source15300、valid510，三seed|相同行/标签；完整缓存torch.equal确认|
|TAM原计数|Bx1x2x1800，80秒规则|Bx2x1800，原计数逐元素完全相同|
|幅值|原计数直接输入|log1p后供固定token与学习卷积；并非新增/减少包预算|
|编码|多层二维/一维卷积、BN、ReLU、层级maxpool/dropout|共享dilation1/3/9的两层16通道GELU生成器，固定30维+学习80维，120token|
|分类与汇聚|类别通道后全局均值|两层Transformer，LayerNorm、位置embedding，均值＋linear|
|优化器|Adam，lr0.0005、coupled wd0.001|AdamW，lr0.001、decoupled wd0.0001|
|LR|按epoch逐段0.0005*0.2**((epoch−1)/30)|6400/9600步处降为0.0003/0.0001|
|训练预算|batch200×30epochs，2310步、459000例次|batch64×12800步、819200例次|
|source扰动|无本项目span_mask|每trace p0.5、双向共同连续90bin遮挡|
|valid选模|20次固定epoch机会|20次固定step机会；同accuracy最早最大|
|历史accuracy/F1|85.948%/85.525%|77.647%/76.889%|

历史RF较高性能并不定位某一瓶颈，因为结构、训练预算、归一化、优化等同时不同。本轮选择两个最小可对照的因素：幅值映射与RF启发优化配方。保留生成器/Transformer/遮挡、batch/步数/索引/选模不变；不移植RF的BN或层级卷积，也不追加增强强度搜索。

RF启发配方保留Adam/lr0.0005/wd0.001，但日程改为按当前固定步数连续0.0005*0.2**((step−1)/12800)。这是优化bundle实验，不是官方配方复制；不能将正/负结果唯一归于Adam、wd或学习率。raw/log1p是同计数的可逆幅值变换（数值有限精度除外），不是新的信息预算；其训练影响仍需实验。即使均阴性，也不能推断BN、层级编码或其它输入变换没有价值。

审计只读取现有source/valid cache和本地代码，不访问未来数据或产生新RF训练/评分。前run已有结果全部保留。机器一致性证据与版本hash见artifacts/preflight.json和freeze.json。
