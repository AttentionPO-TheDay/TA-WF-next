# CPU packet 保序表示实验

状态：frozen，用户于2026-09-26授权CPU首轮实验；已完成结构检查和合成吞吐测量，正式评分前冻结。

问题：固定层次化多视角骨干，保留每个50包patch内完整方向向量是否改善基础分类？只使用既有TemporalDrift source2040/valid510清单，全部source标签。不访问未来日期、WTT-Time、AWF或PCAP；不重新划分已观察数据。

两条件：summary和ordered。均输入102维packet token：[原均值/切换比例2维、有效位置mask50维、方向槽50维]。summary方向槽是均值乘mask，ordered是真实方向（padding为0）。两组都有相同mask/长度信息；run128、window50/250保持原实现。共100个packet token，其他token数量和骨干不变。相同参数量与初始化，但不宣称有效输入秩相同。相较历史模型两组共同新增packet内mask及扩宽投影，因此历史25.235% F1仅作参考，本轮两组间对比才是主要证据。该比较检验包含顺序的原始细节增量，不能唯一归因于顺序（未加入随机顺序对照）。

模型：本项目HierarchicalViewTransformer，d52、4heads、每视角局部1层及跨视角1层，FF104、dropout0.1。仅新增packet_dim参数；无adapter/预训练/蒸馏/TTA。不是CipherSight的TLS record/resource复现。复用本项目生成器与历史抽样/预处理产物，重建全部source/valid原特征核验一致；不加载历史模型权重、不导入旧项目训练代码。显式复用/home/rbf/TA-WF/.venv的Python/PyTorch环境而非其模型实现。

正式预算拟定：两个条件各seeds1729/3407/2026，100epochs，batch64，AdamW lr0.001/wd0.0001，无scheduler，每5epochs eval。source标签只用于训练及训练集诊断；valid只用于按最早最大Macro-F1选择checkpoint和开发比较，无梯度。accuracy同时报告。所有seed从头初始化，相同seed保持相同模型初值和batch顺序。CPU线程/并发数/每seed时间上限依据不看真实性能的合成吞吐在正式运行前确定，最大12 CPU线程、每seed不超过3600秒，最多6个训练seed，不追加超参数搜索。

候选门槛（不是接近DF准入线）：ordered-summary平均valid Macro-F1至少+1.0pp，3/3seed为正，平均accuracy不下降。否则不通过本轮候选门槛。报告全部seed、训练/验证曲线、epoch、预测类别数、耗时及参数量；历史DF49.15% accuracy/48.144% F1只作不同预算性能标尺，不声称严格等预算比较。TemporalDrift仅开发，不称抗漂移或独立确认。

执行安全：入口拒绝draft或非CPU配置，CUDA_VISIBLE_DEVICES为空。每epoch原子保存latest（模型/优化器/随机状态/历史/累计耗时），每次valid提升保存best；达到累计上限按规则停止并保留不完整产物，不能暗中延长预算。评估与checkpoint重载复算独立进行。非有限loss/梯度、数据重叠、散列不符或隔离失败立即停止。所有产物归属此唯一run，完成后自动更新RESULTS/EXPERIMENTS/STATUS。

## 正式资源冻结

合成batch64前向/反向中位耗时：1线程summary/ordered为0.389/0.417秒；2线程0.306/0.304秒；4线程0.245/0.203秒。未进行optimizer更新、未观察真实性能。选择6个worker各2线程（合计12线程），同一seed两个条件相邻启动。估计不含共享资源竞争每seed约17–20分钟；每seed累计训练/评估上限3600秒，在epoch边界检查，最多超一个epoch；监督进程在worker墙钟3780秒时强制停止，防止异常挂起。总计6个seed，最多约12 CPU线程小时，不使用GPU。每seed不足100轮视为未完成，不以较好seed替代。

结构核验5项通过：摘要碰撞在保序表示可区分、梯度有限、partial/padding及batch独立、原层次化模型隔离、含dropout/AdamW及三种RNG的checkpoint恢复逐位一致。既有unittest 43项运行，42通过、1项既有可选parity跳过。pytest未安装，因此按既有unittest及显式函数检查执行，无新增环境依赖。

正式运行：同目录pipeline.py；它启动6个CPU worker，结束后verify_report.py自动复算12组source/valid预测、选模和独立指标，并更新RESULTS/EXPERIMENTS/STATUS。预处理、训练、核验、pipeline及全部本项目src模块的SHA-256记录于artifacts/freeze.json，训练与核验入口拒绝漂移后的代码或配置。
