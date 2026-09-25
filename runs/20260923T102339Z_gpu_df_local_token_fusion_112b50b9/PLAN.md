# 20260923T102339Z_gpu_df_local_token_fusion_112b50b9

状态：frozen；训练前登记。上一轮 `20260923T094615Z_gpu_df_window_fusion_4d785e2e` 的末端扁平 MLP 拼接在 DF 上失败；本轮是由该阴性反馈促成的新机制版本，不是该轮未见确认复试。

问题：与 DF 特征拓扑相配的有序、多尺度 window token 处理器，能否在训练和推理中直接使用生成器并优于同结构常量-token 控制？

输入与权限：复用 TemporalDrift 固定 source2040、valid510、Day14/30/90/150/270各2040行的 sampling manifest。原始 X 仅取前5000方向；本项目 `traffic_views.py` 生成宽50/250的两个有序 window 序列，各 token 为正向比例和转移比例。partial 项设(0.5,0)；全部位置以 full source window 的逐列均值、标准差标准化，标准差低于1e-6设1。source 标签只训练，valid 标签只选最早最高macro-F1 epoch，未来日期标签仅在选模后作已观察开发评分；WTT-Time/AWF不访问。

模型：`df_only` 为迁入的原始 DF。`local_constant`/`local_window` 均在 DF 最后一层局部图 `[B,256,18]` 上接入同一有序 token 处理器。宽50的100-token、宽250的20-token各经独立两层局部 Conv1d(2→32→32,kernel3,pad1,GELU)，自适应平均池化到18位置，拼接后以零初始化1×1 Conv1d(64→256)产生残差；`fused_local = df_local + 0.1*residual`，再走未修改的 DF classifier/head。两个融合条件仅真实/全零标准化token输入不同，参数、初始化、计算、batch顺序严格相同；零初始化使两者在训练前与同 seed 原始 DF 输出一致（preflight验证）。`df_only` 参数较少，仅作为成熟模型参照；首要归因是 `local_window − local_constant`。该实验改变了处理器和融合位置，不是对上一轮处理器单一因素的严格消融。

预算：GPU0，从头训练；3 seed 1729/3407/2026 ×3条件×45 epochs，AdamW lr0.001/weight_decay0.0001、batch64，1800秒上限。各条件仅 valid macro-F1 最早最高 epoch 选模。保存每 seed/角色 accuracy、macro-F1、训练历史、checkpoint、预测、参数和成本。推理仍需生成 token；不加载旧项目 checkpoint、教师或漂移 adapter。

预定判定：`local_window` 相对 `local_constant` 的 valid macro-F1 三 seed 均值>0且至少2/3 seed正，并且五未来日期差均值>0、至少4/5日期均值正，才称 DF 局部 token 候选。另报告其相对原始 DF、上一轮末端融合的绝对表现，但跨轮比较只作描述，不当作严格归因。报告 valid→Day270 降幅，不能只凭绝对F1或降幅之一称抗漂移。window 是方向的确定性变换；收益若有，属于表示/归纳偏置。TemporalDrift 日期已参与方法开发，不构成外部确认。

停止：配置非 frozen、CUDA 不可用、已有 checkpoint/最终预测、source/valid 方向内容重叠、输入或输出非有限、形状/类别错误、超预算则停并保留失败记录。原始数据只读，所有输出归属本 run。
