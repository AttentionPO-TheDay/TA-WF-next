# 20260923T094615Z_gpu_df_window_fusion_4d785e2e

状态：frozen；训练前登记。

问题：保留生成器及 token 专属层参与训练和推理时，DF 主模型能否从 window 方向统计中获益？在更强已有骨干上检验上一轮轻量 CNN 融合的正向结果，而非再做教师—学生蒸馏。

输入：`configs/datasets.json` 中 TemporalDrift；复用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 的 source2040、valid510、Day14/30/90/150/270各2040固定行号。原始有符号时间戳只取前5000方向；由本项目 `traffic_views.py` 生成宽50/250的 window 正向比例、转移比例，partial 项中和为(0.5,0)，再用 full source window 均值/标准差标准化。source 标签训练，valid 标签仅选最早最高 macro-F1 epoch；未来日期标签仅选模后作 TemporalDrift 开发评分。WTT-Time/AWF 不访问。

模型与控制：`df_only` 为本项目迁入的原始 DF(102)；`df_constant` 和 `df_window` 同为 DF feature extractor/classifier（512维）加独立 window MLP(240→128→128)和融合头(640→102)，区别仅标准化 window 输入固定全零或使用真实 token。相同 seed 的 DF 主干初始参数一致；两个融合条件全部参数、初始化、batch 顺序一致。常量分支仍可学习偏置，有效自由度较低；`df_only` 参数更少，因此首要 token 归因比较是 `df_window − df_constant`，对原始 DF 则报告部署收益而不称严格等容量。

训练：三 seed 1729/3407/2026，各条件从头训练45 epochs；AdamW lr0.001、weight decay0.0001、batch64，GPU0，1800秒上限。每 seed/条件独立按 valid macro-F1 最早最大选择 epoch。共同记录 source/valid/五日期 accuracy、macro-F1，训练历史、预测、checkpoint、参数数与耗时。训练及推理都保留 token；不加载旧项目 checkpoint，不更新生成器或漂移适配器。

预定判定：`df_window` 相对 `df_constant` 的 valid macro-F1 均值>0且至少2/3 seed正；五未来日期差均值>0且至少4/5日期均值正，才视为 DF 上的 token 候选。对 `df_only` 同时报绝对收益和漂移衰减。若只胜原始 DF、不胜常量控制，不称 token 特异增益。window 是方向的确定性变换，不能解释为新增原始信息；未来日期已观察，不能称独立确认或单凭绝对 F1 称抗漂移。

停止：配置非 frozen、CUDA 不可用、已有最终预测/checkpoint、源与 valid 方向重叠、类/形状/非有限异常、超预算则停止并保留记录。原始数据只读，所有产物写入本 run。
