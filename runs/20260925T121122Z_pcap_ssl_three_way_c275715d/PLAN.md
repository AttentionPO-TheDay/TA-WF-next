# 20260925T121122Z_pcap_ssl_three_way_c275715d

问题：外部 PCAP 无标签预训练是否改善 packet 或生成器 run/window 表示在少量 TemporalDrift 标签微调下的分类表现？

状态：frozen；这是一次小规模机制筛查，不是最终外部确认。

执行修订（2026-09-25）：首轮主体训练完成，但结果封存路径拼接失败；修正后复测又发现自写 window 算法与正式 `traffic_views.py` 有边界值差异。先前 checkpoint/指标保留为无效实现证据，不据其作方法结论。v2 不改数据、标签预算、模型、训练超参数和选模规则，仅改为正式生成器并修正输出路径，所有产物加 `_v2` 后缀。原始日志和第一次重跑文件有覆盖，已在 RESULTS 记录证据限制。若 v2 失败不得回用 v1 分数。

数据与权限：PCAP 目录只读；每文件最多解析前 30000 个包，按双向五元组分 flow，以首包确定方向，将每 flow 的非重叠片段整理为最多 5000 包序列；短于 1000 包的片段跳过，最多 512 条。文件名类别不进入预训练目标。TemporalDrift 使用既有 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` manifest 的 source/valid，source 每类固定 5 条少标签训练，valid 仅选 epoch；Day14/30/90/150/270 仅在选模后作开发诊断。WTT/AWF 与未来测试不访问。

三条件（完全相同的 packet+window 下游融合结构与少标签清单）：`scratch_fusion`（随机初始化直接监督）、`pcap_packet_pretrain`（仅 packet 分支在 PCAP 上无标签预训练）、`pcap_window_pretrain`（仅生成器 window 分支在 PCAP 上无标签预训练）。两种预训练均对相应输入作 15% 遮挡，预测原始 50/250 window 结构统计；微调时移除重建头并端到端训练融合分类器。这样主模型输入和参数量一致，区分预训练来自哪个分支；但两个预训练任务难度不同，不把差异纯粹归因于生成器。预训练不使用类别、文件名或 TemporalDrift 标签；这是验证外部语料可迁移性的最小目标，不声称完整复现 ET-BERT 的 MBM/SBP。

下游使用 3 个训练 seed、20 epochs、AdamW、valid macro-F1 最早最佳选模。主指标为少标签 valid macro-F1；未来日期只作开发诊断。预算：GPU 0/1/2 各对应一条件并行，预训练 8 epochs、每条件 20 epochs×3 seed、总 900 秒。解析失败、无可用序列、CUDA 不可用或输入非有限立即停止并保留失败记录。预定解释：预训练条件需超过 scratch 的 valid F1 且至少 2/3 seed 正，且五开发日期均值不为负；否则只报告未证实。PCAP 混合应用/VPN 分布与网站指纹目标不匹配、首包方向不一定是客户端方向是主要限制。
