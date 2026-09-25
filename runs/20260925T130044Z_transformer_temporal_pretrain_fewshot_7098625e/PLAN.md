# 20260925T130044Z_transformer_temporal_pretrain_fewshot_7098625e

问题：同域 TemporalDrift 无标签多视角预训练是否改善生成器 Transformer 的少标签微调和跨日期开发表现？

状态：frozen；训练前冻结。TemporalDrift 仅为已观察开发数据，不构成独立确认。

数据与权限：`configs/datasets.json` 的 TemporalDrift；复用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37/artifacts/manifest.json` 已固定 source 2040、valid 510、五日期各 2040 行。source 的 X 可用于无标签预训练；source 每类按 manifest 顺序取最早 5 条作微调（合计 510 条），标签不进入预训练；valid 标签只选最早最佳 epoch。Day14/30/90/150/270 的 X/y 仅在全部 checkpoint 选定后做开发评分，不进入梯度、阈值或训练选择；WTT/AWF 不访问。

输入：仅使用有符号时间戳的符号，固定前 5000 包；正式 `traffic_views.py` 生成 packet、exact run 和 50/250 方向 window；`transformer_proto.py` 将 packet 每 50 包压缩为一个 token，run 最多 128 个、window 120 个，缺失位置显式 mask。无时间数值、包大小、类别/日期身份输入。重建任务共同遮挡同一原始 50 包 span 所覆盖的全部视角 token，预测原始 token 数值；这不是 ET-BERT 字节级 MBM/SBP。

条件（同一模型、相同种子及少标签抽样）：`packet_scratch`（只开放 packet token）、`multiview_scratch`（开放三视角）、`multiview_pretrained`（三视角在 source X 无标签预训练 8 epochs 后微调）。三条件均同结构、同 25 epochs 下游训练、AdamW lr=0.001、batch=64、seed 1729/3407/2026；预训练 batch=64、lr=0.001。由 valid macro-F1 选择最早最佳 epoch，每 seed/条件同 25 次机会。主比较是 pretrained vs multiview scratch；packet scratch 是生成视角绝对增量的辅助对照，不声称等有效容量。

主指标：valid macro-F1；开发日期分别报告绝对 macro-F1、accuracy 和 valid→Day270 下降。预定候选门槛：pretrained−multiview scratch 的 valid 平均 >0 且至少 2/3 seed 正，五开发日期差均值至少 4/5 日期 >0。若不满足，仅报告未证实。多视角相对 packet 的结果单独列出；与既往 DF 强基线并列解释，不把胜过同一小 Transformer 当成部署胜利。

预算与停止：GPU 0 一块，最多 8×3 预训练 epochs + 25×3×3 微调 epochs，batch=64，整轮最多 2400 秒；保留每 seed/条件最佳 checkpoint、历史、预测和指标。预处理或输入异常、CUDA 不可用、source/valid 方向重复、运行超时、非有限损失、数据/配置不符即停止并保留失败记录。原始数据只读。TTA 不在本轮执行；以后另立 transductive 权限与固定阈值/步数协议。
