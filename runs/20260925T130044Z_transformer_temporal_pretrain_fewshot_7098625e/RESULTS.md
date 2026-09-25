# 实验结果

状态：completed。TemporalDrift 已观察开发数据，非独立确认；没有执行 TTA，也没有访问 WTT/AWF。

## 主要结果

相同 102 类、每类 5 条 source 标签、三训练 seed、25 次 valid 选模机会下，Macro-F1 百分比均值：

| 角色 | packet scratch | 多视角 scratch | 多视角无标签预训练 |
|---|---:|---:|---:|
| valid | 3.732 | 2.625 | 4.511 |
| Day14 | 2.422 | 2.279 | 4.196 |
| Day30 | 2.821 | 2.066 | 3.728 |
| Day90 | 2.741 | 2.068 | 3.656 |
| Day150 | 2.449 | 1.499 | 3.348 |
| Day270 | 2.178 | 1.721 | 2.918 |

无标签预训练相对同结构多视角 scratch 的 valid 增量为 +1.886 pp，三个 seed 均为正（+1.015/+2.483/+2.160 pp），五开发日期均值也全部为正，通过预定相对候选门槛。相对 packet scratch 的 valid 增量 +0.779 pp，三个 seed 均为正；五开发日期的绝对 F1 也均高于 packet scratch。因此，这一小型 Transformer 上的同域无标签预训练得到**相对正向开发信号**。

但绝对表现很弱。预训练 valid F1 只有 4.511%，Day270 2.918%；既往同一 5-shot 清单上的轻量 CNN scratch fusion 曾达到 valid 20.298%，虽然架构和训练方案不同，足以提醒当前模型还没有达到可用的主模型水平。多视角 scratch 甚至弱于 packet scratch。预训练的 valid→Day270 下降为 1.593 pp，packet scratch 为 1.554 pp；本轮不能称为减缓漂移衰减。

## 收敛和解释

三个条件的最佳 epoch 大多在 21–24/25，末轮训练准确率仅约 3.5–12.3%；说明固定 25 epochs 下尚未充分拟合，当前低绝对 F1 不能归咎于“Transformer 不适合”。无标签重建 loss 三 seed 首轮 0.248/0.215/0.225，末轮 0.155/0.154/0.154，证明优化目标被学习，但不等于网站语义或抗漂移机制已学成。

当前只使用 TemporalDrift source 2040 条 X 预训练，没有使用外部 PCAP；source 每类 5 条标签微调，valid 仅选最早最佳 checkpoint。五开发日期的 X/y 在所有九个 checkpoint 完成 valid 选择后才读取；query 不进入训练或适应。三条件同一 110,400 参数模型，packet-only 仅关闭 run/window 输入，有效自由度不严格相同。无需教师—packet 学生蒸馏；预训练编码器直接进入微调和推理。

## 核验与版本

`artifacts/integrity.json`：63 组预测、checkpoint 重载、指标与最早最佳 epoch 重算，0 错误；每类少标签数 5，source 无标签预训练数 2040，预处理产物仅含 source/valid。source/valid 方向完全重叠在预处理时拒绝（本轮通过）。原始数据只读；WTT/AWF 未访问。

代码 SHA-256：`transformer_proto.py` `ba921e57111694effd9df4b00e3da7adf3b538a931c047a15f64d1626fc868b6`；本 run `train.py` `9163b2ac47fc6a66ef015091f1496c22cce9753851d5d46325ec768de74c2d29`；`config.json` `3b57f743a177ac44feeceea8dae468ca18b514f59e582ee1444177dc7882763e`；固定 sampling manifest `f03b6fa0047d1daff655851ea5bfcc76f3973552a934e272ca9f1899a1c2f40b`。

下一步不宜立即在低置信度教师上做 TTA；先另立预算匹配的收敛检查或更强编码器对照。若看到结果后延长 epochs，须登记新版本，不能将本轮小 valid 当作独立确认。
