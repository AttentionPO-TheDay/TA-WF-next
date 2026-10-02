# 渐进packet CNN→Transformer与同CNN对照

授权：2026-10-01用户“开始实验吧，可以并行进行”，承接渐进CNN→Transformer单视角建议。本轮6个新训练任务、同一GPU最多2并发，另有实现/核验并行协作。未来数据不授权访问。

## 问题及比较

H1：渐进降采样packet生成器与Transformer能否超过既有多视角Transformer源期表现？H2：同生成器/读出下跨token attention相对逐token残差MLP是否产生稳定增益？不预设输入压缩为唯一瓶颈。

A：方向5000→三阶段CNN，每阶段两层kernel7 stride1卷积、逐位置LayerNorm/GELU、kernel4 stride4 ceil mask maxpool，通道32/64/128，序列5000/1250/313/79。每层显式mask、无效池化位置负无穷、空输出归零。79×128 token加可学习位置嵌入→2层pre-norm Transformer、4头、FF256、dropout .1→LayerNorm及masked mean→102类。
B：相同CNN/位置/末端norm/读出，将2层Transformer替换为2层逐tokenpre-norm残差MLP（FF256、dropout .1）。两条件共享部分初始化逐元素相同，抽样流相同，端到端CE训练。A512262、B379654参数，共同CNN223776；不是严格参数匹配，不能唯一归因于attention。

历史对照：显式复用 `20261001T041301Z_gpu_source_fit_native_baseline_ee3c7640` 的GPU三seed Transformer（F1 50.881%）和DF参照（69.643%），不重新训练、不加载checkpoint初始化、不视为独立重复。历史Transformer同数据/输入信息、步数、优化器、LR和20次选模；新结构同时改变表示和容量，不能单因子归因。DF配方、参数量及计算不同，仅性能参照。旧CPU54.248% accuracy不替代同GPU对照。

## 数据和权限

遵循 configs/datasets.json 的TemporalDrift定位，显式复用本项目已审计缓存 `20260927T032819Z_source_size_local_depth_factorial_dae714b7/artifacts/prepared.pt` 和 manifest.json。使用source150（15300，102×150）及原valid510（102×5），不重划分。仅方向sign(float32前5000 signed timestamp)，padding为0，无时间/包大小/URL/未来信息。source标签只用于CE与获准诊断；valid仅预定checkpoint选择与指标核验，不梯度/不适应。本轮不添加run/window、预训练、蒸馏、TTA，不读取任何未来日期或WTT/AWF。固定valid已反复观察，所有结论为开发证据。

## 训练、裁决和预算

三seed21729/23407/22026，两条件从头初始化；每seed都使用seed+9000+cycle的randperm循环，15300样本拼接后切分batch64，12800步共819200样本呈现。AdamW lr .001、weight_decay .0001；第6401步降至.0003、第9601步降至.0001。全FP32确定性计算，不TF32/AMP；CPU2线程每worker。

valid仅步3680,4160,...,12800共20次，最高Macro-F1取最早checkpoint。报告相同checkpoint下source与valid accuracy/F1、last、参数数与时间；不会中途追加/延长训练。预定比较A−B、A−历史Transformer、A−历史DF、B−历史Transformer：平均F1增益>=1pp、3/3seed正且平均accuracy不降为候选通过；不是统计显著性结论，只有一次样本抽样。H1/H2分别报告，不能以胜弱对照替代胜DF。

单物理GPU0，GPU1/2由其他任务占用不触碰；GPU0最多2任务并发，同seed A/B优先。每任务3600秒，上层另60秒退出宽限；整批14400秒，supervisor强制终止所有子进程组；任何OOM、非有限loss、核验失败、超时停止新增任务并保留产物，不自动降低batch重试。

## 实现来源和核验

model.py本项目本轮独立实现。worker.py基于上一轮run内训练器明确最小改写（旧训练循环非旧项目隐式依赖）；仅sys.path引入本项目src，旧venv仅提供第三方包。所有新增代码、配置、计划、输入和manifest、历史配置/report/预测文件冻结SHA-256，不修改旧run。

test_model.py四组CPU合成检查：token形状和mask、被mask垃圾/NaN隔离、batch独立性、全padding输出及梯度有限、各卷积非零梯度、同seed共享初始化和RNG一致。preflight额外检查真实source CUDA batch64反向与batch128推理（不优化、不valid评分），核验source/valid方向重复为0及类别数量；显存估计每worker额外预留2GiB，两个worker合计小于可用显存85%才启动。

每任务保存独立日志、initial state、index stream、first gradients、history、best/latest checkpoint、best/last预测及report。重建模型加载best、逐条预测一致；sklearn独立复算两角色指标。supervisor最终核验两条件共享生成器/position/readout/final_norm初始化相同和批次流相同，复算12份历史预测指标；统一更新RESULTS.md、STATUS.md及EXPERIMENTS.csv。所有产物归本run，不覆盖历史结论。
