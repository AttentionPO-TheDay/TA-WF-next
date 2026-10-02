# wide+BN窄优化配方与创新性审查 v1

用户授权“开始下一步实验”并要求调查当前设计创新性。实验目标：固定wide+BN逐级生成器＋Transformer结构（当前开发均值89.150%），只比较训练配方能否缩小到90%；创新性结论单独写入RESULTS，不把高分直接称原创。

历史baseline复用上一run wide_bn三seed。新条件lr05（初始/分段lr .0005/.00015/.00005，wd.0001）、wd05（原lr .001/.0003/.0001，wd.0005）、lr05_wd05（两者同时），各三seed共9新任务。结构、初始化、索引、span_mask、batch、步数、20次valid选模、CE、source/valid权限完全固定。不加Mixup/teacher/新输入。AdamW betas.9/.999 eps1e-8。GPU0最多3路，单任务3600秒，总18000秒；其他GPU/CPU训练关闭。

模型精确复用wide+BN：逐级32? 本run的wide为64/128/256三stage，6个BatchNorm1d，stage池化3/5/1，80维学习特征+30维固定log-count，Transformer d128 2层/4heads/FF256/dropout.1。所有BN统计仅source train更新，eval统计冻结；best counter及buffer完整核验。当前源数据TemporalDrift source15300/valid510，valid仅选模/指标；未来/WTT/AWF/适应关闭。

门槛：相对baseline平均accuracy≥1pp、3/3正、F1不降；90%独立判断。新条件若均未达，不继续扫优化配方；冻结wide+BN后转向跨日期/外部协议审计。每条valid约.196pp，目标差.850pp，实验只提供开发证据。

创新性审查范围：复核本项目2026-09文献矩阵、RESEARCH_ASSESSMENT与literature_failure_crosswalk，识别已有WF CNN/Transformer、稳定表示、时频/对比/少样本/漂移适应、BN/元学习/自监督重构近邻。结论分为已知组件、组合工程差异、尚未形成贡献。有限本地文献审查不能推出全领域无先例；需要未来定题时定向查重。当前唯一较有潜力的新问题是“有序burst条件关系遮蔽预测用于漂移适应”，但已有TTT/NetTTT/NetCLR/WF-TFC近邻，尚未验证，不在本实验训练。
