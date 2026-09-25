# 20260923T043131Z_cpu_window_tail_adapter_12d25c24

问题：在固定 TemporalDrift 抽样上，window 适配器是否改善 valid 与未来开发日期泛化，且性能是否依赖 partial/tail 信息？

数据：configs/datasets.json 指向 Proteus TemporalDrift；复用 20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37 封存的 source/valid/Day14/30/90/150/270 行号，不重新抽样。source 仅训练，valid 仅 macro-F1 选 epoch，未来日期只作开发评分，不作梯度或适配。

条件：window_full；window_adapter（零初始化 32 维残差适配器，训练时与分类头联合优化）；window_tail_neutral（partial window 的正向比例置 0.5、transition 置 0，固定语义后重新按 source 统计标准化）。三条件共享 240 维 window 编码、优化器、batch、epochs、训练 seeds。

指标：每个 seed/条件记录 source train accuracy/loss、valid accuracy/macro-F1、Day14/30/90/150/270 accuracy/macro-F1、最佳 epoch、partial window 计数。主判定是 valid 不下降且多数未来日期、平均未来 macro-F1 不下降；tail-neutral 大幅下降只解释为长度/尾部依赖，不能称抗漂移。

预算与停止：CPU 4 threads，15 epochs，3 seeds；GPU/WTT-Time/AWF 关闭。若输入重叠、非有限值、配置非 frozen 或已有预测则停止。未来日期结果属于 TemporalDrift 开发证据，不能当独立确认。
