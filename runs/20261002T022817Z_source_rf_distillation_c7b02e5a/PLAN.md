# 源期 RF 蒸馏 v1

问题：固定保留的77.647%生成器＋Transformer，源期RF软目标是否改善验证准确率？用户授权尝试蒸馏。

条件：CE-only三seed历史复用；KD lambda0.1、0.5；shuffled lambda0.5（固定独立source行乱序，保留目标边际分布）。T=2，CE不缩减，KL方向teacher||student，batchmean乘T²。每条件21729/23407/22026；9新任务。主比较两KD对CE，打乱对照作知识归因诊断；不追加调参。

教师：显式复用本项目native RF run同seed历史best，曾按valid选择epoch30/26/30，并非未经valid反馈教师。训练输入clean raw TAM，软目标来自教师训练过的source15300，非out-of-fold。学生训练既有source span_mask，形成clean-teacher→masked-student目标。只导出source logits/rows/labels，禁止valid教师软目标进入梯度。教师不训练，学生从头初始化，推理只保留学生，不做集成。原RF训练3×30epochs计作额外历史成本，不称总算力匹配纯CE。

数据固定TemporalDrift source15300/valid510；configs/datasets.json已定位，复制已冻结prepared缓存，不重划分。source标签训练，valid仅20次选模与评价；未来/WTT/AWF/适应关闭。

学生原flat_none两层16通道共享dilation1/3/9生成器、TAM log1p＋Transformer、原AdamW、span_mask；不得采用未通过门槛的归一化/层级模块。batch64/12800步，3680起每480步共20选模，最大valid accuracy最早严格改善。索引seed+9000+cycle、mask seed+50000，乱序目标seed+70000。所有新任务共享对应seed历史初始化和索引。

门槛：三seed平均accuracy增益≥1pp、3/3正且平均F1不下降；≥90%单独报告。多条件及已观察valid仅开发证据，不保证提升、独立泛化或抗漂移。

预算：GPU0最多3并发，float32无AMP/TF32，2CPU线程/worker，每任务3600秒，整批18000秒。CPU只准备和审计。训练前检查同row/label/TAM、完整教师source logits重载（误差容限1e-4，预测一致）、历史学生预测和初始化一致、KL方向/T²/detach、三路内存。冻结代码/配置/数据/目标/历史checkpoint hash；所有失败保留，不超预算补跑。
