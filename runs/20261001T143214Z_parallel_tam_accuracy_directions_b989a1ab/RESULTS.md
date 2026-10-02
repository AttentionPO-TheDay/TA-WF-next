# TAM并行方向实验结果

准备完成，尚未评分正式训练结果。24项新训练（18 GPU＋6 CPU）与3项历史基线，数据、方法、预算及源码已冻结。7项机制检查、CPU/GPU真实source预检、三任务显存预算及6组历史重载预测核验通过。CPU卷积对照为本轮CPU Transformer，避免直接把设备数值差异归因于结构。

机器进度见artifacts/progress.json，任务日志位于logs。仅source/valid开发，无未来日期或适应。

## GPU阶段结果与部分完成后复核（2026-10-01 15:28 UTC）

核对时18项GPU新训练已完成；CPU新训练仅完成cpu_temporal_cnn seed21729，另两项CPU baseline正在运行、三项CPU任务排队。总管与两CPU进程仍存活，状态保持running。不能将用户报告“实验完成”直接作为整批完成依据；CPU结构比较必须等全部三seed与同设备baseline完成。

67个冻结文件hash一致。已完成22份报告（18新GPU＋1新CPU＋3历史baseline）的best/last×source/valid共88组accuracy/F1独立复算通过，初始化、索引、12800步、20次验证及最早最佳选模规则核验通过。已有worker及历史审计核验checkpoint重载预测；本次仅读取已有缓存/预测复算，无新训练或未来访问。证据见artifacts/partial_postrun_audit.json，整批完成后由原总管生成全批结果。

|GPU条件|accuracy三seed均值|Macro-F1|相对baseline accuracy增量|正seed数|增量门槛|
|---|---:|---:|---:|---:|---|
|baseline历史复用|75.948%|75.295%|—|—|对照|
|scale_concat尺度拼接融合|76.144%|75.289%|+0.196pp|2/3|未过|
|attention_readout注意力读出|75.098%|74.423%|−0.850pp|1/3|未过|
|relative_time相对时间|75.033%|74.345%|−0.915pp|1/3|未过|
|span_mask连续遮挡增强|77.647%|76.889%|+1.699pp|3/3|通过|
|supcon监督对比损失|75.817%|75.103%|−0.131pp|2/3|未过|
|cosine_lr余弦学习率|73.922%|73.066%|−2.026pp|0/3|未过|

GPU阶段仅span_mask通过预定开发门槛。逐seed accuracy77.647/78.235/77.059%，配对增量+0.980/+2.745/+1.373pp，平均Macro-F1增量+1.594pp。GPU最高均值距90%仍差12.353pp。保留它为GPU阶段开发候选，不在CPU未完成时宣布整批最佳或自动拼接其他改动。

span_mask仅训练时每trace概率0.5遮掉两方向同一90bin区间，评价输入完整；best checkpoint平均source accuracy97.440%，低于baseline98.887%，valid更高，支持这一具体训练增强配置的源期开发收益，不唯一证明过拟合来源或已解决泛化。其三seed共同错误53条与baseline相同，尚有稳定难例。所有GPU候选末轮valid低于best，本轮不支持直接延长训练。其余五GPU方向保留阴性结果，不继续扫描强度/融合/温度/LR；不能将其推广为所有相对时间、attention或对比学习均无效。

已完成的CPU卷积seed21729 valid accuracy75.294%、Macro-F1 74.325%，仅单seed且对应CPU baseline尚未完成，暂不下结构比较结论。原CPU队列与预算保持不变，等其完成后再按预定规则判断整体候选。已观察510 valid反复开发、七候选筛查与三优化seed不构成独立确认；无抗漂移或外部泛化结论。

执行中止，部分产物保留：
Traceback (most recent call last):
  File "/home/rbf/TA-WF-next/runs/20261001T143214Z_parallel_tam_accuracy_directions_b989a1ab/supervisor.py", line 101, in main
    time.sleep(5)
  File "/home/rbf/TA-WF-next/runs/20261001T143214Z_parallel_tam_accuracy_directions_b989a1ab/supervisor.py", line 80, in <lambda>
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(RuntimeError('supervisor terminated')))
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rbf/TA-WF-next/runs/20261001T143214Z_parallel_tam_accuracy_directions_b989a1ab/supervisor.py", line 80, in <genexpr>
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(RuntimeError('supervisor terminated')))
RuntimeError: supervisor terminated

## 用户授权设备切换（2026-10-01）

用户要求CPU任务切换GPU。原总管按SIGTERM停止并保留19项已完成新训练、CPU部分checkpoint、日志及全部原冻结文件；总管通用handler曾登记failed，现更正为stopped，原因是用户授权设备切换而非算法失败。为避免跨设备续训改变随机数/数值路径及不对称对照，不把CPU未完成模型直接续训混成原重复；新run 20261001T154039Z_gpu_tam_temporal_cnn_b0a9af83在GPU0从头训练同卷积设计3seed，对照复用已完成GPU Transformer baseline。原CPU CNN单seed75.294%保留为描述性开发结果，不混入新GPU均值。不追加其它候选、未来权限或valid机会；新run独立冻结设备/预算及既往结果暴露。
