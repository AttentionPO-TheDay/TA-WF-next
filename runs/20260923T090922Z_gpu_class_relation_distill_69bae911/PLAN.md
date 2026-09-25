# 20260923T090922Z_gpu_class_relation_distill_69bae911

问题：冻结融合教师的source类别原型关系能否通过训练期蒸馏，使packet-only学生优于普通CE和同教师packet分支关系对照？

状态：frozen，训练前登记。

问题：逐维特征 MSE 和完整教师 logit 蒸馏未稳定受益；本轮改为教师分支的 source 类别原型相似关系，检验类别相关结构能否传给只用 packet 的学生。方法和控制在训练前固定，不以未来日期调节。

数据与权限：通过 `configs/datasets.json` 读取 TemporalDrift，复用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 中 source 2040、valid 510、五未来日期各 2040 的固定行号。只取前5000个方向；source 标签用于学生 CE、类别原型与教师关系目标，valid 标签只选最早最佳 epoch，未来标签仅选模后开发评分。输入 feature 来自本项目 `20260923T090023Z_frozen_feature_transfer_probe_18dd81c9/artifacts/features.npz`：其三seed融合教师 packet/window 128维 source 特征已按 source 每维统计标准化，checkpoint/生成规则/抽样/结果核验完毕。显式登记 feature artifact SHA-256，不加载旧项目 checkpoint。教师特征只在训练 source 中使用；学生推理不读取 token、原型、教师或投影。

关系目标：对同一融合教师的 packet 或 window 分支分别构建 102 个 source 类别原型（每类20样本均值），L2 归一化。source 样本与各原型的余弦相似度乘固定 scale=10，再 softmax 成 102 类关系分布。对样本自身类别使用排除该样本的19例原型；学生训练时计算对应类别相似度也使用相同排除版本，防止自身纳入原型。原型只用 source 特征和标签，无 valid/未来信息。

学生：同一 packet-only CNN、128维隐藏层及102类分类头；训练期另有线性128→128投影，仅用于对齐教师类别关系，推理时丢弃。`ce`、`relation_packet`、`relation_window` 三条件逐seed共享初始化、batch顺序、AdamW lr0.001/weight decay0.0001、45 epochs和valid选模。CE仅主分类损失；两关系条件均为 `CE + 0.5*KL(教师关系分布 || 学生关系分布)`，且原型数、维度、温度/scale、损失和可训练参数完全相同。CE条件实例化但不使用训练期投影，不声称等有效容量；两关系条件严格匹配。

门槛：`relation_window` 同时相对 CE 与 `relation_packet` 的 valid macro-F1 均值>0、至少2/3 seed正，并在五个未来日期平均差>0、至少4/5日期均值正，才视为训练期 window 特异候选。报告全部seed、valid/日期accuracy和F1、原型关系目标的 source 真类 top1/熵、训练损失/准确率、最佳epoch与成本。若只训练拟合或仅优于CE不优于packet关系控制，不成立。

资源/停止：GPU0单卡，3seed×3学生×45epochs，1200秒上限；配置非frozen、CUDA不可用、旧feature或抽样散列不符、已有checkpoint/最终预测、方向source/valid重叠、形状/类别/非有限异常则停止。原始数据只读。TemporalDrift 是已观察方法开发数据；WTT-Time/AWF保持关闭。
