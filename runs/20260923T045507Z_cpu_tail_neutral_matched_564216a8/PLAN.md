# 20260923T045507Z_cpu_tail_neutral_matched_564216a8

问题：tail-neutral 在完全共享 full source 标准化统计与模型容量下是否仍改善 TemporalDrift 泛化？

数据和抽样：复用 20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37 的封存 source/valid/Day14/30/90/150/270 行号；source 训练，valid 选模，未来日期只作开发评分。

条件：window_full 与 window_tail_neutral。两者使用同一 240 维模型、训练 seeds、batch、15 epochs；唯一差异是 partial window 是否置为 (0.5, 0)。两条件均使用 full source 的 mean/std，避免条件特定标准化造成混杂。

判定：valid macro-F1 不下降，且未来日期平均及多数日期不下降，才认为 tail-neutral 有初步泛化证据。所有未来结果仍属于 TemporalDrift 开发证据，不是独立确认。
