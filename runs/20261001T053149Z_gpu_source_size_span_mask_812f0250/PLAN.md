# CNN-MLP源期样本量×局部遮挡

授权：用户2026-10-01请求下一步实验，承接错误分析、数据量审计及样本量×合理增强建议。前置审计run `20261001T052912Z_source_error_capacity_audit_b32edd4f`已完成；source合规18553条，比source150增加3253（21.26%），每类156–190，中位184。固定valid510中短流量错误更多，MLP三seed共同错83条；六个模型共同错54条；简单多数投票仅74.31%，不足以解释90%目标。

## 有界方案

2×2条件：source150×无增强（复用上一run三seed历史MLP，不重复训练）；source150×遮挡、全部合规source×无增强、全部合规source×遮挡（三条件各三seed，共9新任务）。仅CNN-MLP，不增加Transformer或多视角。不预设90%可达。

模型完整复用渐进packet run `20261001T044328Z_gpu_progressive_packet_transformer_b7ba43ae/model.py`，源文件hash冻结；所有权重从头初始化，不载历史checkpoint训练。与历史baseline共享初始化及source150抽样流须最终逐元素/散列核验。固定seeds21729/23407/22026，batch64，12800步，AdamW .001、weight_decay .0001，6401/9601步lr降为.0003/.0001，dropout .1。固定20次valid选模（3680起每480步至12800），最高Macro-F1取最早，accuracy辅指标及90%目标单独报告。source标签只CE及诊断，valid标签只选模和评分，不梯度不适应。

局部遮挡：仅训练，每条概率.5；在有效前缀内随机选一个连续span，长度floor(有效长度×.02)，限制1..32包且至少保留1包；direction置0但原observed mask保持为真，表示人为缺失方向而非padding，包索引不压缩、不移位、不改顺序；padding保持不变。独立augmentation RNG seed+700000，不改变batch流。推理完全无增强。这是可能保留类别语义的弱扰动假设，未经证明模拟真实流量变化；不进行强裁剪或大幅丢包，阴性结果照常报告。

数据经configs/datasets.json定位，读取前置审计产出的prepared.pt和manifest.json（源train.npz与full valid.npz方向去重已检查，删除168条valid重复、718条source重复，无标签冲突）。source150嵌套保留，固定valid510不变化；未使用的valid样本不加入训练。all采用18553行均匀shuffle，每类156–190，存在轻微不均衡，故与150相比同时改变样本量、类频率及固定步数下每样本遍历次数，不能唯一归因于数量。相同样本规模的增强开关共享逐步batch。source评价分别基于各自训练集，不能当相同样本拟合对比。

## 预定裁决、资源与核验

比较150增强−150无增强、all无增强−150无增强、all增强−all无增强、all增强−150增强、all增强−150无增强；平均F1>=1pp且3seed均正、平均accuracy不降为候选门槛，非统计显著性。一次抽样及已观察小valid只构成开发结果。不增加valid查看次数、不依据中途曲线调整增强强度或训练步数。

GPU0最多2任务并发，每worker2CPU线程，每任务3600秒（外层60秒宽限），整批4小时。GPU1/2保持不使用。OOM、非有限loss、超时或核验失败停止并保留产物，无自动缩batch/调参重跑。supervisor负责进程组退出、总预算、RESULTS/STATUS/索引。

训练前CPU检查遮挡长度、非破坏性、prefix/padding、确定性、空/单包输入；GPU synthetic前向/反向与batch128推理检查。已有同模型显存预检每worker<1GiB，额外数据仍在此前2GiB安全预留范围。运行时固定FP32确定性，无AMP/TF32。

最终每任务新模型重载best checkpoint并核验source/valid逐条预测一致；sklearn独立复算accuracy/102类Macro-F1。9任务18份新预测，3历史任务6份指标复核；共享initial state及同样本规模batch流核验。所有实现/config/PLAN/输入/历史引用散列冻结。模型、worker与supervisor是显式本项目run代码复用；第三方依赖来自既有venv，不隐式导入旧项目训练代码。无未来日期/WTT/AWF、无TTA/预训练/蒸馏。
