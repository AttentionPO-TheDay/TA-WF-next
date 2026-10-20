# 多因素诊断子集评测规范 v1

冻结日期：2026-10-19（Asia/Shanghai）。执行 run：`20261018T164924Z_multifactor_diagnostic_v1_4fc8f13d`。本规范根据用户明确授权，将已观察公开数据解锁为 **Diagnostic Evaluation**；不改变 PROTOCOL.md 的独立确认标准，不回写或否定历史门槛审计。

## 数据、身份与任务

唯一底座为仅用 TemporalDrift/train 的 18,553 条保守候选训练的三个 wide_bn checkpoint（42/100/2026）。源 valid 已用于选模，不是独立 Day0 测试。底座与源 Token 统计量、102 类原型直接复用已审计产物，不重新训练。

Version：从四份 drift 的逐行恢复清单提取 status=unique 的行，按 attributed_version 分为 045/046/047/048，按完整 trace group 去重；同一 trace 只保留字典序最早 file,row。四份 drift 共用的 10,017 条未知 trace 全部剔除，绝不按父目录推定其版本。清单并集原有 90,145 条明确版本 trace，全部是版本 train/valid 锚点重用；本实验未用这些版本锚点训练底座，但它们不是独立新采集、也不代表完整原始 drift。045→048 差异是诊断关联，不作渲染优化或版本升级的单因素因果效应。

Network：DE/JP/SG/UK/USA 为 Cross-Region Geographic Slices。仅报告地域切片间性能离散度，不赋予未记录的丢包率、RTT、jitter 或路由原因。地区不是某一物理变量的随机干预。

Behavior：原始 20,400 行、17,739 个精确 URL，先每 URL 取 raw_group_id 最小的一个 trace（平手按行），不把 20,400 行称作 17,739 个独立网站。用 urlsplit.hostname 的小写值、去末尾点与前缀 www. 作为弱分组；hostname 不是顶级域名或 eTLD+1，不推定不同子域一定是同一首页。URL/hostname 只用于分组隔离与事后 NMI/ARI、分组指标，不输入适应损失或模型。任务表述为“深层 URL 交互切片诊断”，不从 F1 变化直接推断背景噪声机制。

跨家族分类采用冻结的 **内容支持标签对应假设**：既有 cross_family_label_evidence.csv 在三个家族分别支持完整 102 对一一对应标签，来自相同原始 trace 的历史审计，不由本次 query 预测拟合。报告原始 102 类集 Macro-F1/accuracy，明确不是官方 hostname↔homepage 身份验证；该标签假设可能不覆盖未共享内容的语义。本轮允许其用于诊断评分，不扩大到官方身份或独立确认声明。Behavior 另报预测分区与 hostname 的 NMI/ARI，作为无需把模型类别命名为 hostname 的补充。不给 query 做最佳标签置换/Hungarian 匹配。

## 角色隔离与输入

所有行先排除纯净 source 与选模 valid 的有效 TAM 精确碰撞、全零/非有限输入及目标之间重复有效 TAM，按固定切片顺序保留首条，记录每项删减、类别覆盖、row/file/group/URL/hostname。输入严格复用纯净底座：原始前 5,000 个有符号时间戳转 float32；绝对时间按 80 秒裁剪为 2×1,800 TAM；不排序、不差分、不伪造 IAT。保留原始输入及有效 TAM 散列。不能以精确隔离证明 session 独立。

每切片有效行按 SHA256("diagnostic-v1|"+raw_group_id) 排序，首 1,024 为无标签 reference，其余为 query；Behavior 排序键改为完整原始 URL，并先执行 URL 去重，因此 URL 也隔离。不得因类别覆盖或评分好坏重抽。reference 标签不进入优化 API，query 输入不参与任何适应、统计量估计、选模或阈值选择。query 标签只用于固定终点评分及预定可视化。每切片独立从纯净 checkpoint 重置。

TENT BN 仿经典方法只更新 BN affine、移除源 running statistics；query 前向仍使用批统计量，列为单独的 transductive-forward 对照。其他条件只从 reference 更新，query 固定状态。该信息权限差异在表注中保留。

## 方法与有界预算

三 seed、十切片、七条件共 210 个终端评分单元：frozen、tent_bn、tent_entropy_gate、token_adain、token_projector、prototype_no_topology、prototype_anchor。Gate 只更新 Transformer 内八个 LN affine，锁定源 BN/final_norm/classifier。AdaIN/projector 复用上一版定义。新增原型方法保留相同 110→16→110、残差系数 0.1 的恒等初始化瓶颈，隔离目标函数变化；底座参数与缓冲区全部锁定，不额外引入 LoRA。

原型方法仅使用冻结 source 原型和独立 reference 的纯净 teacher 预测：H≤1 nat 且最大概率≥0.8 才作硬分组，不读取 reference 真标签。目标类中心经 source 标准差归一化后，与对应 source 原型作距离拉近；当批中至少两个类各有≥2个可信样本，约束中心两两余弦距离与源类间关系一致。损失为窄门控熵 +0.1×中心距离 +1×类间关系 +1×源 Token 恒等保持 +0.001×残差。`prototype_no_topology` 仅把关系项权重设零，其他完全一致，用于辨别拓扑项的贡献。所有归一化参数只来自 18,553 行；source anchor 固定为按源行散列选取的 1,024 行。不把教师错误的聚类视作真实类别。详细公式与实现冻结在 mechanisms.py。

所有优化固定 800 步，Adam lr=0.00025、batch=128、无 dropout；新方法裁剪梯度范数 1.0。最终步唯一 checkpoint，不按 query 或探针选择；不扫描超参数，失败/阴性同样报告。GPU0、GPU2 两队列，三 seed 总预算≤6 GPUh、wall≤3h、单元≤900s、输出≤8GiB。预检只用 source，原始数据只读，全部产物归属 run。代码、配置、规范、依赖审计和数据清单在首次目标评分前 SHA256 冻结；修错必须留痕。

## 指标、图表与判据

Table1 给出十切片 frozen 三 seed 均值±样本SD、源 valid 描述性保留率、Version 048−045 差值、Network 极差/SD、Behavior 相对源 valid 差值。源 valid 与目标难度不匹配，保留率不能当成因果破坏权重。雷达图使用原始 F1 及三因素等权均值，不拟合“物理权重”。

Table2 全报七条件、逐切片与因素等权汇总的 F1、accuracy、ΔF1、NAR=P(适应错|frozen对)、Recovery=P(适应对|frozen错)。主候选诊断成功需三个因素等权平均 ΔF1>0、相对 gate ΔF1>0 且 NAR≤gate；拓扑消融需 full 相对 no_topology 的均值 ΔF1>0 且 NAR 不增加；均不保证通过，也不称零误杀。

Table3 只复用已经完成且独立审计的外部 v4 全部 144 单元，含负收益条件与 WTT 2022-02-16 异常切片。这检验各外部历史数据独立训练方向模型的方法迁移，不是 Temporal 闭集分类器跨数据集零样本识别，也不是新原型方法的外部证据。已读取的 WTT/AWF 结果影响本轮选题，明确登记 post-external-feedback 新版本；本轮不重跑外部测试。

Figure1/2：固定 seed42、Version048 和 Behavior、数值标签0–9每类按 group 散列最多40条query，源同类最多40条；类缺失如实报告。提取 WideGenerator.time 在 time_projection 前的 pooled 110D Token。source/original/aligned 在同一 source 标准化空间中一起拟合单次 t-SNE、共享坐标显示前后，固定随机种子与超参数；不分别旋转/挑选 embedding 制造靠拢。真实标签只供事后着色。另报告原空间配对原型距离、最近原型正确率、类间距和方差塌缩指标。t-SNE 是辅助图，不能独立证明流形恢复，未改善时保留原图。
