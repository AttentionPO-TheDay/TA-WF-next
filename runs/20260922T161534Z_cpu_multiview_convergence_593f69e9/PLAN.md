# 20260922T161534Z_cpu_multiview_convergence_593f69e9

问题：统一45epochs后run/window融合和适配器结论是否保持

状态：frozen，训练前冻结。用户授权沿上轮结论做统一训练预算检查，无正在运行或已完成的同一任务。

沿用20260922T114658Z_cpu_run_window_adapters_v2_148f23cd全部五模型及初始化/数据/优化，不改变结构，只从15统一延长至45epochs。显式复制其train.py和完整audit.py（后者作为verify.py），不隐式导入旧代码，不加载checkpoint（原run未保存末轮优化器状态，不能拿best权重冒充继续原轨迹）。同seeds1729/3407/2026从头运行，前15轮是复现锚点而非独立新证据；只额外保存15/30/45累计best及末轮预测。学习率始终0.001、AdamW wd0.0001、batch128。

source2040/valid510沿用token-native封存清单，原数据根configs/datasets.json，共102类，每类20/5。source标签梯度训练，valid只选模/开发评分；未来/WTT/AWF关闭，不适应。窗口50/250方向比例及转向比例，共240；source固定位置均值标准差归一化（std<1e-6置1），valid复用。run前512方向/桶token，观察预算5000，无时间/资源/精确长度旁路。生成器冻结，模型和适配器联合训练，推理保留，不是训练期蒸馏或漂移更新。

模型与前轮完全一致：run_pair两run编码器；window_pair两window编码器；fusion各一支无adapter；fusion_shared共享32→16→32零初始化残差adapter；fusion_specific各自32→8→32。每支输出512，concat1024→102分类，实际参数153766–158470，约3.1%差。容量近似而不是严格等算力，shared/specific瓶颈宽度不同。run在512run处截断而windows覆盖前5000观察，覆盖不同，结论限定当前接口。

预算5条件×3seed×45epochs，CPU3线程，GPU隐藏，总timeout1200秒，预计约10分钟；无网格或追加seed，非有限/hash错/隔离错/超时停止保留。选模valid macro-F1最高，平局最早，全部45轮都运行；15/30/45只是同一路径三个累计选模预算，不是独立重复。选模机会增加会使累计best不降，不能据此单独证明泛化或收敛改善。

主比较45轮specific-fusion（适配器增量）、fusion-window_pair和specific-window_pair（相对强基线）。沿用上轮门槛：均值F1差>0且至少2/3seed F1差>0作为候选，不是显著性；同时报告accuracy、shared/specific、全部逐seed与预算曲线。只有融合同时超过两单视角对照才称当前预算下互补候选。比较15→30→45累计best、固定末轮F1、loss/训练accuracy和最佳epoch；最佳在36–45仍标为预算末段，避免声称严格收敛。报告平均最后5轮训练loss相对第31–35轮变化作描述，不事后调参。

执行前合成维度/mask/输入权限/初始恒等/两步梯度测试；训练中封存配置/脚本/输入，checkpoint全valid重载；结束独立重建2550输入及source-only归一化、隔离、指标、45epoch选模、预算预测、hash核验。valid反复用于开发，无独立确认、抗漂移或生成器最终增益主张。失败阴性均登记。
