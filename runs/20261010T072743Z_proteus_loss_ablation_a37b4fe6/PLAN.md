# Proteus 损失组件消融 v1（2026-10-10）

问题：上一轮移植在 Day14 退化、Day90/270 改善且负适应较大，哪些损失项在当前配方中贡献了这种现象？先前将其归因为 GMM 的表述没有消融证据，本轮只检验条件性贡献。

## 条件与公平控制

六个条件：full_control（全部四项）；no_pseudo（去伪标签CE）；no_mmd；no_entropy_diversity（同时去样本熵与边际熵）；no_source_ce；source_ce_only。最后一项只保留 source CE 损失，仍有目标和伪标签三路前向及 BN 更新，因此不是完全没有目标域信息的 source-only 训练。

所有条件：当前项目 wide+BN 三个 checkpoint，同一初始权重、Adam lr .001、100 epoch、batch128、每 epoch8步，共800次更新；每5 epoch刷新两分量 GMM、tol1e-6、可靠 posterior≥.6。每个日期独立从 source checkpoint 重置，不累计跨日期适应。全部模型可训练，BN以训练模式更新，dropout开启。

三路前向（source、target、pseudo）在所有条件保留，损失权重设0进行消融。故 no_pseudo 测试的是伪标签监督梯度的边际贡献，不是彻底关闭 GMM 或目标统计的效果。GMM划分随模型更新而变化，这是机制反馈的一部分。

为保证 source/target 采样不被伪标签池大小改变，三个 loader 使用独立 RNG；每次更新重置 dropout RNG。同一 seed/date 下，source/target行序与dropout随机数相同。此调度相对上一轮端口有明确变更，因此新训练 full_control 作为同轮归因基准；历史 Proteus只作参考，不要求与历史结果精确相等。公式和GMM float64归一化沿用上一轮端口，不再称原论文端到端严格复现。

## 数据、标签与选模

仅 TemporalDrift Day14/90/270（已观察开发集）、seed21729/23407/22026。逐行核对原1024条support和其余query，精确输入指纹隔离。source回放仅用原训练部分，valid不加入回放。进入适应的support对象仅含输入和行号，标签不传给适应函数。query输入不进入适应，所有条件只在第100 epoch结束后预测与评分；无query选步、早停、调阈值或适应。未来外部数据与其他因素不开放。configs/datasets.json和prepared数据只读。

## 记录、预算与裁决

独立run登记，54个新任务，每个保存checkpoint、logits、loss轨迹、GMM覆盖率、抽样流hash、更新数、参数与BN改变量和耗时。历史36份预测明确标记复用，不算新重复。

单GPU0进程、CPU2线程，整批6小时硬上限，单任务30分钟硬上限。出现OOM、非有限值、伪标签不足128条、输入hash变化、split交叉、梯度异常或审计失败立即停止保留记录；不自动调参重跑。可在未评分前做合成公式/梯度一致性及显存检查。

主比较每个单项删除相对同轮full_control的固定102类Macro-F1、accuracy、负适应率、纠错率；逐日期和全部9个配对单元全部报告。预注册开发信号要求F1平均+0.5pp、负适应率下降≥1pp、至少6/9单元F1提高。该门槛是后续候选筛查，不是统计显著性结论；不从单次消融断言一个组件是唯一原因。source_ce_only用于检验继续source拟合和BN更新背景。与10步adapter的计算预算不同，不能将其性能差异当同预算方法比较。

执行前核对输入/代码/config/PLAN来源hash；执行后独立复算指标、抽样配对、checkpoint重载预测、损失和梯度更新权限。总结自动更新RESULTS.md、EXPERIMENTS.csv及STATUS.md当前事项。
