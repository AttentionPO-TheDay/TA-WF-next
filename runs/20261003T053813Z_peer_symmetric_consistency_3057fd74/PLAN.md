# 单网络互蒸馏与一致性对照

## 假设和实际机制

在线 EMA 配方未改善主网络；本轮去掉固定教师和 EMA，使用同一模型的两视图预测相互约束。两个分支共享全部参数且均接受 source CE 监督，对方温度概率作为 stop-gradient 目标：

`L = (CE(z_a,y)+CE(z_b,y))/2 + w(t)*T^2*(KL(stopgrad(p_a)||p_b)+KL(stopgrad(p_b)||p_a))/2`。

无额外模型、教师参数、EMA 状态、历史软目标或 checkpoint 初始化。这是单网络视图互蒸馏，不声称是新颖算法或已验证漂移机制。

## 对照和可解释比较

- `dual_mask_ce`：两个独立 span-mask 视图，只做两视图平均 CE。
- `mask_peer`：相同两个 mask 视图 + 对称互蒸馏。
- `dual_cross_ce`：clean 与 span-mask 两视图，只做两视图平均 CE。
- `cross_peer`：相同 clean/mask 两视图 + 对称互蒸馏。
- `ce_only`：显式复用上一轮三 seed 的单视图 CE 产物；不重复训练，属于历史参考。

主要归因比较：mask_peer 对 dual_mask_ce、cross_peer 对 dual_cross_ce。与历史单视图 CE 比较判断绝对收益，不能把双视图监督的收益归因于一致性。每个新条件运行 seed 21729、23407、22026，共 12 新训练 + 3 历史参考。对照与对应 peer 条件匹配初始化、索引、增强随机流、双视图前向和 dropout 随机过程。

## 前向和 BN

两个视图拼为 batch128，单一模型执行一次前向，拆为各64的 logits；联合 BN 统计且每 optimizer step 只更新一次 BN，避免顺序前向造成 BN 双更新。历史单视图参考为 batch64，计算量和 BN 样本数不同，故机制归因只用匹配双视图 CE 控制。

## 冻结参数和数据权限

T=2，w 在前3200步为0，3201至6400步线性升至0.25，此后保持；用 CE 预热防止训练初期相互放大随机预测，不设置信度门控。不根据 valid 调整。

复用冻结 TemporalDrift source15300/valid510、102类 prepared；共享原始数据根见 configs/datasets.json。本轮不读新的原始数据。signed timestamps5000、TAM2x1800、log1p、mask p0.5且90bin；两个mask流为seed+50000/seed+60000。原timestamp在两视图相同。source标签用于CE；valid标签仅选模/评价，不入梯度。未来日期、WTT-Time、AWF、TTA全部关闭。

wide+BN64/128/256，d_model128、Transformer2层4head、FF256、dropout0.1；batch64独立样本、12800 optimizer steps、AdamW lr0.001/wd0.0001、6401/9601步lr降为0.0003/0.0001。20个固定选模步点继承config，最高valid accuracy、并列最早；报告accuracy和Macro-F1。训练全从头初始化。

## 一致性诊断和判定

best checkpoint另在同一个valid的预定mask(seed91919)上只读推理，记录masked accuracy、clean-mask预测一致率、T=1概率平均JS散度；不用于选模、阈值或损失调整。历史CE使用本地复制的checkpoint只读诊断，绝不用于初始化。

候选对匹配双视图CE需accuracy均值至少+1pp、3/3seed正、F1均值不下降；同样记录对历史CE是否达到该门槛以及绝对90%。一致率增加不能单独视为分类提升，更不能视为抗漂移。

## 预算与停止

GPU0最多2路，12新任务各7200秒、总43200秒；FP32、确定性、无AMP/TF32。损失非有限、梯度/target异常、冻结hash不符、checkpoint重载不一致或预算超限停止并保留全部失败产物。监督程序自动汇总并复核，STATUS与EXPERIMENTS保持一致；不频繁轮询。开发valid已多轮观察，不是独立确认。
