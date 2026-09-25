# 20260923T075107Z_gpu_packet_window_main_378247f0

问题：生成器window token经专属处理层加入packet主模型后，能否超过同容量packet-only及window-only并在TemporalDrift日期上保持泛化？

状态：frozen，训练前登记。

问题：生成器输出的 window token 经独立 MLP 编码后，与 packet 编码融合训练，是否优于同容量的 packet 单视角和 window 单视角？本轮明确 token 分支在训练和推理时都保留；不代表训练期蒸馏或漂移后适配器更新。

输入：`configs/datasets.json` 中 TemporalDrift；复用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37/artifacts/manifest.json` 的 source/valid/Day14/30/90/150/270 固定行号。各样本前 5000 个有符号时间戳只取方向；window 由本项目 `traffic_views.py` 生成，50/250 宽，partial 项中和为 (0.5, 0)，再使用原始 full source window 均值/标准差。source 标签仅训练，valid 选模；未来日期标签仅在选模后用于开发评分。WTT-Time/AWF 不访问。

同结构条件：`packet_only` 使用真实 packet 和常量 window；`window_only` 使用常量 packet 和真实 window；`fusion` 两者均真实。三者完全相同的卷积 packet 分支、window 专属 MLP 和 256→102 分类头、参数数目、初始化 seed、batch 顺序、优化器及评估机会。常量输入为全零（相应表示空间的 source mean/padding 中心），分支仍执行；因此该对照是等总参数和算力，但单视角条件的常量分支可训练部分有效自由度较低，不声称严格等有效容量。window 是 packet 的确定性变换，任何收益是编码归纳偏置而非新增原始信息。

预算：GPU 0 一块，3 seed（1729/3407/2026）×3条件×45 epochs，AdamW lr=0.001、weight decay=0.0001、batch=128；1200 秒上限。仅 valid macro-F1 选 earliest-best epoch，不以未来分数挑模型。记录每 seed/日期 accuracy、macro-F1、source train loss/accuracy、best epoch、参数量与时间。

预定判定：主候选要求 `fusion-packet_only` 在 valid 平均 F1 >0、至少 2/3 seed 正，且五个未来日期平均 F1 差 >0、至少 4/5 日期均值正；互补性另要求 `fusion-window_only` 满足同门槛。若仅训练集拟合更好，或融合不及最强单视角，不称生成器带来可靠主模型增益。valid→Day270 下降只作漂移诊断，不替代绝对 F1。外部数据集在冻结方法前不评分。

停止：配置非 frozen、CUDA 不可用、已有最终预测/模型、抽样数量或类数不符、源与 valid 方向重叠、非有限值时停；若超时保留失败记录。原始数据只读。
