# 分段汇聚对照

授权：用户2026-10-01要求开始下一步，承接保留粗粒度位置的分段汇聚建议。

问题：当前CNN-MLP+局部遮挡已有valid accuracy73.987%、F1 73.394%，全局均值是否丢失可用粗粒度位置？不宣称90%可达、不把该假设视为已定位原因。

两个新条件×三个seed：G将全局masked mean复制四份；S按有效token的相对顺序分成四段，分别masked mean后按顺序拼接。二者都输入512维线性102类头，总参数相同。token rank r属于floor(4r/n)段，n为有效token数；空段为零，空流量为零表示。相对有效长度分段而非固定5000四等份，避免短流量多数段为空；不伪称资源或burst边界。

历史H显式复用20261001T053149Z_gpu_source_size_span_mask_812f0250的s150_mask三seed，不重新训练、不加载checkpoint初始化。模型/增强/训练器显式复制该run最小改写，无旧项目隐式依赖。新head在原模块初始化完成后用fork_rng构建，每段权重初始化为原128维head权重/4，bias不变。两个新条件所有权重初值逐元素一致，与历史backbone也一致。G初始函数与历史H数值一致，但扩大参数化改变优化，且重复输入有效秩低，不能宣称有效容量完全匹配。要求S同时超过G和H才支持保序汇聚成为部署候选；G优于H只表示优化参数化变化有作用，不支持位置机制。

数据沿用configs/datasets.json所指TemporalDrift经source_error_capacity_audit生成的prepared.pt/manifest：只source150(15300)和既定valid510，原raw数据只读、不重划分。方向前5000包，无时间/包大小/URL；source标签用于CE和诊断，valid仅20次选择与获准核验，禁止梯度/适应。缓存source_all不用于训练。未来日期、WTT/AWF、预训练、TTA均关闭。

固定CNN三阶段+两层逐tokenMLP、dropout .1、每类150条、已有局部遮挡：训练每条概率.5，连续floor(2%有效长度)方向置零，上限32包、至少保留1包，observed不改变，独立RNG seed+700000；推理无增强。三个seed21729/23407/22026，从头初始化。每seed batch流沿用seed+9000+cycle，batch64，12800步。AdamW .001，weight_decay .0001，第6401/9601步降至.0003/.0001，FP32确定性、无AMP/TF32。valid步3680起每480步至12800共20次，Macro-F1最大取最早，报告同checkpoint source/valid及last。

预定比较S−G、S−H、G−H：平均F1>=1pp且3seed全正、平均accuracy不降为候选通过，不是统计显著性判断。只有一次训练样本清单及已观察valid，不能当独立确认或抗漂移证据；不依据中途曲线加候选或延长。

资源：物理GPU0最多2任务并发，共6任务，GPU1/2不使用。每worker2CPU线程、单任务3600秒外层60秒宽限、整批14400秒硬上限；OOM、非有限loss、核验失败或超时终止保留产物，不自动重试调参。仅使用旧venv第三方依赖。

预检：手算相对四分段参考、空序列、padding垃圾隔离、batch独立、两条件所有权重一致、历史backbone一致、G与H初始函数近似一致；synthetic GPU梯度及batch128推理显存。正式训练前freeze全部源码/config/PLAN/输入/manifest及引用历史模型/预测/report/初始化/batch流hash。每任务best模型新实例重载逐条预测核验，sklearn独立指标复算；最终12份新预测及6份历史指标复核、初始化及batch流一致性核验。supervisor更新RESULTS/STATUS/实验索引，所有产物归本run。
