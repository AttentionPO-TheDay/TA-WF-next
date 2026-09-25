# 20260923T122727Z_gpu_varcnn_window_fusion_f65c399f

状态：frozen；训练前登记。

问题：以本项目已有 `VarCNNDirection` 为 packet-direction 主干，直接加入生成器的有序多尺度 window token，能否同时超过原始 VarCNNDirection 与同结构常量 token 控制？

输入和权限：复用固定 TemporalDrift sampling manifest 的 source2040、valid510、Day14/30/90/150/270各2040行。X只取前5000方向；window为宽50/250的正向比例与转移比例，partial置(0.5,0)，按 full source 逐列均值/标准差标准化。source标签仅训练，valid仅最早最高macro-F1选epoch，未来标签仅选模后开发评分；WTT-Time/AWF关闭。

模型：`varcnn_only`为本项目已迁入 `VarCNNDirection(102)`；`varcnn_constant`与`varcnn_window`均为同一方向主干及512维特征，宽50的100-token序列、宽250的20-token序列分别经独立保序Conv1d(2→32→64，kernel3、padding1、ReLU)再全局平均池化，各64维，拼接为128维。该token特征与512维方向特征拼接后经线性640→512、ReLU、Dropout0.5及102类头。两融合条件仅输入全零/真实window不同，参数、初始化、batch顺序、优化和选模机会相同；`varcnn_only`作为强原始主干参照。该版先使用序列内局部编码与全局融合，不声称位置对齐已被验证。

训练：GPU0，seed1729/3407/2026×三条件×45epochs，AdamW lr0.001、weight decay0.0001、batch64，1800秒上限。从头训练，不加载旧checkpoint、教师或adapter；训练和推理均保留window token。保存逐角色accuracy/macro-F1、历史、checkpoint、预测、参数和耗时。

预定准入：`varcnn_window−varcnn_constant` valid均值>0、至少2/3 seed正，且五日期至少4/5均值正；同时 `varcnn_window−varcnn_only` valid均值≥+0.5pp、至少2/3 seed正，五日期至少4/5均值正且平均≥+0.5pp。任一失败都不能称VarCNN部署候选。valid→Day270下降只作诊断，不称抗漂移。

停止：配置非frozen、CUDA不可用、已有最终产物、source/valid方向重复、输入/输出非有限、形状/类别错误或超预算则停并保留记录。原始数据只读，产物仅写入本run。
