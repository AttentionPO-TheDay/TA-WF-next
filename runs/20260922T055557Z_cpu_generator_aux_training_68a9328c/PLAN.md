# 20260922T055557Z_cpu_generator_aux_training_68a9328c

问题：相同packet推理模型下window或run辅助训练是否改善source辨别力：三seed小规模CPU对照

状态：frozen，真实数组访问前冻结。用户明确生成器仅辅助训练，漂移缓解由其他模块承担。本轮只测试一种辅助回归训练实现。

数据：复用TemporalDrift CPU v2 manifest中的source2040/valid510行号，不新划分；source已排除完整valid方向重复并canonical去重。本轮只读取train/valid，未来日期/WTT/AWF不读。配置通过configs/datasets.json定位数据。source标签训练，valid标签每epoch选模，故valid是开发选择结果不是独立测试。

三条件baseline/window_aux/run_aux，各训练seed1729/3407/2026，9次从头训练。共用packet小CNN：Conv1d1-16 k9 s4/ReLU/Conv16-32 k7 s4/ReLU/AdaptiveAvgPool32/Linear102。主模型初始化和batch顺序配对一致，推理仅packet，108326参数。所有条件均构造相同辅助头Linear1024-240和Linear1024-8；baseline不反传辅助损失。训练参数额外开销与推理参数区分报告。

window目标为生成器50/250窗口的positive_fraction和transition_fraction，共240固定槽，缺失窗填0；不添加显式长度列。run统计8维：每方向分别run数量、平均长度、长度标准差、最大长度，对四值log1p；无该方向则0。使用完整5000观察内所有run，不使用512截断。辅助目标只从source生成，逐维source标准化（std<1e-6置1）；packet亦source逐位置标准化。目标完全由packet信息派生，不增加新模态。

CE分类损失 + 固定0.1 × 辅助逐元素均值MSE；无联合双辅助、调权重或搜索。AdamW lr0.001 wd0.0001，batch128，15epochs，CPU4线程，CUDA_VISIBLE_DEVICES为空，timeout600秒。每epoch按valid macro-F1最大选checkpoint（平局较早）；所有条件都有15次选择机会。冻结预测后汇总accuracy/macro-F1及相对baseline三seed配对差，保存模型、源统计、目标、history、输入/代码hash。

预定候选判读：某辅助条件的valid macro-F1三个seed差均正且accuracy均值差正，列为后续扩大验证候选；否则称混合或未获一致收益。此门槛是开发筛查，不是显著性或未见确认，不以抗漂移为生成器准入标准。记录辅助loss是否下降；负结果只针对这个目标/权重/预算，不否定生成器。

源码独立实现，packet架构和标准化沿用本项目TemporalDrift v2，生成器来自src/ta_wf_next/traffic_views.py；借用旧venv依赖，不引入旧训练代码/checkpoint。非法值、样本隔离错误、超时则停止；不根据结果延长预算。
