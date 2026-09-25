# 20260922T011309Z_token_effect_diagnostic_d0b9386f

问题：比较原始方向、精确run、粗化run及多视角表示的信息保留与跨条件稳定性；首轮只做冻结统计诊断，不训练评分

状态：frozen CPU pilot，2026-09-22。原创建说明的“仅统计、不评分”已在任何本轮数据访问或结果产生前修订为CPU轻量读出；原因是用户请求测试作用且GPU三卡98–100%忙碌。不以纯接口诊断替代效用评价。

问题：token视角是否保留可由相同简单读出利用的网站信息？此为原型读出开发实验，不是深度模型公平性能定论、分级路由或TTA效果证明。

源：NetworkDrift/train，每类20条；source validation：NetworkDrift/valid，每类5条。目标开发条件：NetworkDrift/JP每类最多20条（不足取全部）、BehaviorDrift/subpage每类20条。固定seed1729、按类排序后Generator抽样；target y仅用于预先声明的分层抽样和最终评分，不进入features、prototype、尺度或融合。它不是部署阶段按类采样方案，只是诊断数据设计。仅一个源子集，不扫描seed/阈值/超参。

每trace前5000原始观测，无排序。六个预先固定系统：packet；exact-run log1p(count) signed序列；coarse-run signed(1+floor(log2 count))；50/250窗口方向比例和转向密度（无显式长度通道）；同方向interval与run span的log1p秒值+有效掩码；全部五视角cosine分数等权均值。前两run序列保留全部run到5000槽，不固定少量token截断；末run可能截断保留。时间view先计算同方向间隔与run跨度，无混合IAT，无rate；mask区别0与缺失。

各视角相同读出：只用source估计逐维均值/标准差（std<1e-6按1处理），逐trace L2归一，按真实source类均值建prototype再L2归一，cosine打分。所有超参固定；source valid仅报告，不用于选模。不同输入维度/归纳偏置明确披露，不声称严格匹配深网容量或优化预算；无可训练encoder。融合无调权重，无自适应门控/蒸馏。实现runner在本run，不修改生产生成器。

预算：单进程CPU/BLAS2线程，零GPU、零神经网络训练/optimizer/checkpoint，source prototype估计不是“没有任何拟合”。采样行与输入内容哈希保存，train/valid/JP/subpage完整内容与方向前缀交叉复核。已知Network/Behavior共享source不能算两次独立训练；subpage不跨开发/确认拆分，不声称未见URL泛化。

所有配置预测先保存封存，再用对应标签评分accuracy、macro-F1、逐类accuracy与视角错误互补；报告全部结果，不只报最优。禁止根据JP/subpage反馈返调本轮。JP和subpage正式登记为性能开发已暴露，剩余Network国家、Version、Temporal、WTT/AWF均不加载或评分。即使负结果也只否定本读出实现，不代表token学习机制无效。
