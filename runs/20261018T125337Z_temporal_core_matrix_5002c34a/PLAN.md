# 论文一：纯净源域时序核心矩阵与动力学采集

2026-10-18用户明确发布三阶段。阶段1已完成且weight_audit通过，复用18,553条TemporalDrift/train的三seed纯净权重（42/100/2026），不重训、不选最好seed、不加载旧权重。源valid是已观察选模指标，不称独立Day0 test。9项固定query的纯净frozen基线已完整完成，按hash及行号复用；不合并早前中断CPU批次，不把引用当成新重复。

阶段2主表为3seed×Day14/90/270×5条件共45单元：9项既有frozen，36项新适应；另增加9项tent_coadapt纯熵全参诊断，总计54指标/45个新适应checkpoint。严格复用既定1024无标签reference及独立query，不重划分。source replay只用锁定18,553行；query标签只在terminal800后评分。原始数据只读，所有输出留在run中。

TENT BN只更新6个BN的affine、移除running统计量，适应与query均使用各自批统计量。query无梯度但有batch-statistics信息，这是单列声明的transductive-forward对照。tent_entropy_gate仅四个Transformer LayerNorm affine，源BN参数和缓冲区以及final_norm/classifier均锁定，0.2–0.4 nat gate。Proteus-default移植sourceCE+GMM硬伪标签CE+Gaussian MMD+entropy/diversity，lr0.001=1.0x，GMM2/tol1e-6/阈值0.6/每40步刷新；不是官方RF端到端复现。no_pseudo_fixed只用sourceCE+MMD+entropy/diversity，lr0.00025=0.25x，锁定final_norm/classifier，其他参数和BN统计按source/target正常更新。所有Adam、batch128、terminal800，不追调。该联合对比不单独归因某个组件。

tent_coadapt为诊断额外对照：全参无门控目标熵，Adam0.00025，源BN缓冲区固定。图1对比它和gate，scope与门控同时改变，不能据此单独作Scale-Lock因果归因。0–800每10步在同一1024 reference上eval固定probe，保存logitL2、H<0.2占比、分类器与final_norm范数，probe不改BN或选模，训练batch另逐步记录。图2只用终端Day270 query输出计算熵；实测H<0.2比例，不预设98%或低熵必错。图3记录NAR=原正确变错/原正确数、Recovery=原错误变正确/原错误数，分母0记null，不预设谁最低；冻结不适应的NAR零值作基点。

主表每日期3seed均值±样本标准差：固定102类Macro-F1、Accuracy、配对ΔF1(pp)、NAR、Recovery；逐单元预测、权重和日志全保留。匹配source/target loader种子2025/2026，逐更新dropout种子102024+step；GMM及pseudo loader固定seed。参数/缓冲区作用域、来源/输入hash、模型feature包装一致性先经source-only smoke验证。适应后独立重载45 checkpoint，sklearn复算54条指标，并核对probe步数和锁定集合。

GPU0/1/2每卡一个seed worker，运行至多2小时墙钟（workers最多6 GPU小时），每单元2400秒，输出4GiB；前置预检额外≤600秒、独立审计≤1800秒、渲染≤300秒。NaN、锁定/数据隔离/hash断言失败、GMM退化或伪标签不足即停止保留失败；不静默降强度/换模型。完成单元仅在预测和checkpoint hash均符时跳过，未完成从同一纯净权重重跑，不新增选模机会。

Temporal日期及既有外部反馈均已观察；本次是方法开发与机理测量。后续若据此改法，必须登记新版本及外部暴露，不消除已观察测试反馈。渲染阶段单独登记在配套figure run，按冻结脚本输出PDF/SVG/PNG，不通过调训练制造预设图形。
