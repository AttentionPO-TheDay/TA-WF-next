# 20260925T135213Z_transformer_cpu_convergence_07cd94a5

问题：上一轮 25 epoch 的低分类分数是否主要由监督训练未收敛造成？固定表示与模型，把监督微调延长到 100 epoch，分别观察训练集拟合和 valid 泛化；不更改 token、run 截断、预训练目标或优化器。

状态：frozen。先前 25 epoch 与 valid/TemporalDrift 开发日期结果已经观察，本轮是明确的新开发实验，不是独立确认或旧实验补写。

数据：复用上一轮仅含 TemporalDrift source/valid 的 `prepared.pt` 和固定每类 5-shot 清单（510 条标签），记录 SHA-256。source 全 2040 条 X 可用于多视角无标签预训练；仅 510 条 source 标签进入 CE 微调。valid 510 条标签仅用于预定 checkpoint 选择及开发评价。五未来日期、WTT/AWF 均不访问；不执行 TTA。

条件：`packet_scratch`、`multiview_scratch`、`multiview_pretrained`。沿用上一轮 110400 参数模型、输入规则、三 seeds 1729/3407/2026、batch 64、AdamW lr 0.001/weight decay 0.0001；pretrained 条件仍从头在 source X 预训练 8 epoch。全部条件重新开始，监督训练 100 epoch；不接续或加载旧 checkpoint。

测量：每 5 epoch 在固定 510 条训练样本和 valid 上以 eval mode 分别计算 accuracy/Macro-F1；另记录每 epoch CE loss 与训练模式 batch accuracy，后者不代替独立训练集评估。各条件/seed 用第 5、10、…、100 轮 valid Macro-F1 选最早最佳模型（20 次机会相同）。报告 25/50/100 epoch 的 train/valid 指标、最优 epoch 及类别预测覆盖。与旧 25 轮结果只作背景比较，不视作等选模机会的严格配对。

预定解释：若 train accuracy 到 100 epoch 仍低于 50%，优先查表示、分类头和优化；若 train 高而 valid 仍低，则优先查少样本泛化/输入压缩；若 train/valid 均明显提高，下一步再做预算匹配的表示消融。以上是诊断分流而非性能成功门槛；只有 valid 与既有强 CNN/DF 和跨日期指标足够好，才讨论部署/抗漂移。不得事后只挑有利 seed。

资源：仅 CPU，最多 4 PyTorch 线程；三条件×三 seed×100 监督 epoch，预训练三 seed×8 epoch；总 wall-time 上限 7200 秒。每完成一个 seed/条件即原子保存 history、指标、checkpoint、预测；超时或异常保留已有记录并标记 incomplete。原始数据只读，输出仅在本 run。停止条件：非有限 loss、数据/代码 SHA 不符、资源上限到达。
