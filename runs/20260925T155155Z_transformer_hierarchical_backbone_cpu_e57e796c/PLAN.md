# 20260925T155155Z_transformer_hierarchical_backbone_cpu_e57e796c

问题：仅把平铺多视角 Transformer 改为视角内局部编码、视角摘要与跨视角全局编码，是否改善固定 TemporalDrift 5-shot 的基础分类？这是受 CipherSight L1/L2 层次化设计启发的视角级类比，不是论文的 TLS-record/flow 实现，也没有 resource 语义标签。

状态：frozen。TemporalDrift source/valid 已反复用于开发；本轮是新结构的开发筛查，不是独立确认。只检验主模型结构；不改变 packet 50 包摘要、run 128 截断、window token、优化器、标签预算，不使用无标签预训练、adapter、LoRA、P1/P2、蒸馏或 TTA。

数据与权限：复用 `20260925T135213Z_transformer_cpu_convergence_07cd94a5/artifacts/prepared.pt` 的固定 source2040/valid510 和每类5条 source 标签（共510），SHA-256 `4dc489bf088246c59b73cdb071bb22459c975051ed84b27962bbbf0067d70b24`。source 5-shot 标签仅用于 CE 训练；valid 标签仅按预定指标选模和开发评分。其它 source 标签、五未来日期、WTT/AWF 不用于训练、适应、选模或评分。原始数据只读。

方法：三个视角各有独立 1 层 Transformer 局部编码器及 View-CLS，之后将三个视角摘要送入 1 层全局 Transformer + Page-CLS，线性分类。输入仍是正式生成器产物。`d_model=52`、4 头、FFN=104，参数113670。对照为先前 `20260925T135213Z_transformer_cpu_convergence_07cd94a5` 已完成的平铺 `multiview_scratch`（110400 参数），明确复用其结果而非本轮新增独立重复；两者容量差约3%，不能声称严格等容量。改变的不只是注意力拓扑，也包含分视角独立参数和位置编码，因此仅能归因于整套层次化编码方案。

预算与选模：新模型从头训练 seed 1729/3407/2026，每 seed 100 epochs、batch64、AdamW lr0.001/weight_decay0.0001、CPU最多4线程，总 wall-time 上限7200秒。每5轮以 eval mode 对固定510训练样本和510 valid 计算 accuracy/Macro-F1；用 valid Macro-F1 在5,10,…,100轮选最早最佳，和已完成平铺对照同20次机会。保存逐轮曲线、最佳 checkpoint、train/valid 预测与指标。若任务超时或异常，保留已完成 seed 并标记 incomplete，不挑有利 seed。

主判定：层次化 valid Macro-F1 相对平铺对照的三 seed 均值差>0、至少2/3 seed正，且层次化 train eval accuracy 不低于平铺对照均值，才保留为基础模型候选；否则记录阴性。另报告与既往同5-shot轻量CNN 20.298% valid F1的描述性差距。DF 48.144% 由全部source标签训练，标签预算不同，不能用作本轮公平达标门槛。无未来日期评分，任何结果都不证明抗漂移。
