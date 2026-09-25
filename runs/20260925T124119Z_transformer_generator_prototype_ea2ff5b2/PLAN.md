# 20260925T124119Z_transformer_generator_prototype_ea2ff5b2

问题：固定多视角生成器加小型 Transformer 能否跑通无标签预训练、少标签微调和教师约束 adapter-only TTA 的结构闭环？

状态：frozen；仅做 CPU 合成结构检查，不进行真实数据训练或漂移评分。

范围：`src/ta_wf_next/transformer_proto.py` 接收 `traffic_views.py` 的 packet/run/window token；每种 token 有独立输入投影和 type embedding，Transformer 输出 CLS 分类表示。无标签目标是跨视角共同遮挡同一原始 packet span 后的 token 重建；少标签阶段使用显式 labels 的 CE；TTA 阶段冻结 teacher 和 student backbone，仅更新 bottleneck adapter，以高置信 teacher KL 加同流量双视角 cosine consistency。

权限与限制：本轮不读取 TemporalDrift、PCAP、WTT 或 AWF，不使用类别标签，不创建新数据划分。TTA helper 不接收 query 标签，也不代表允许对保留测试集适应。真实训练前仍需另立冻结实验，预先规定无标签数据角色、扰动、阈值、步数、回滚和评价日期。

验收：CPU unittest 覆盖 typed batch、forward、batch 独立性、mask span 不泄露、重建梯度、少标签 step、adapter identity、teacher confidence selection 和 TTA 更新隔离；所有输出有限且 shape 正确。完成实现时增加完整 5000 包预算和 adapter 参数实际更新检查，不改变任何真实数据或模型训练权限。失败即停止，不报告模型收益。
