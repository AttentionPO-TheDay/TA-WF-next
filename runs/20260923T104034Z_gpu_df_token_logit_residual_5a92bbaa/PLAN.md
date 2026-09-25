# 20260923T104034Z_gpu_df_token_logit_residual_5a92bbaa

状态：frozen；训练前登记。上一轮 `20260923T102339Z_gpu_df_local_token_fusion_112b50b9` 的真实 token 相对等容量常量分支有效，但尚未超过原始 DF；本轮据该反馈提出新版本，不能将同一 TemporalDrift valid/日期当作未见确认。

问题：保留原始 DF 分类路径，仅让生成 token 学习小幅类别分数修正，能否取得相对原始 DF 的净收益，且收益不是新增 packet-only 容量造成？

数据与权限：从 `configs/datasets.json` 定位 TemporalDrift，共享只读数据；复用固定 sampling manifest 的 source2040、valid510、Day14/30/90/150/270各2040行。X 仅取前5000方向。由本项目 `traffic_views.py` 输出宽50/250的有序 window 方向正向/转移比例；partial 项中和(0.5,0)，以 full source window 逐列均值/标准差标准化（std<1e-6设1）。source 标签梯度训练，valid 只按最早最高macro-F1选epoch，未来日期仅选模后开发评分；WTT-Time/AWF不访问。

模型和控制：`df_only` 为本项目迁入原始 DF(102)。`residual_constant` 与 `residual_window` 均保留相同 DF feature extractor/classifier/head；100与20个window token分别经独立保序 Conv1d(2→32→32,k3,pad1,GELU)及全局平均池化为各32维，再与DF 512维分类特征拼接，经线性576→128、ReLU、零初始化线性128→102得到修正 logits。总 logits=`df_logits + 0.1*correction`；初始化与同 seed 原始 DF 输出相同。两个残差条件只在标准化 token 为全零/真实方面不同；参数、初始化、batch顺序、优化及选模完全匹配。其常量分支可利用DF特征学习额外 packet-only 修正，是严格的新增容量控制；`df_only` 参数较少，是部署基线。上一轮在 DF 18位置局部图加残差，本轮只在 logits 加修正，因而改变位置及读出方式，不将跨轮结果归因于单一因素。

训练：GPU0，从头训练；seed1729/3407/2026×三条件×45 epochs，AdamW lr0.001/weight_decay0.0001，batch64，1800秒上限。训练和推理均使用 token；不使用旧项目checkpoint、教师蒸馏或漂移adapter。记录 source/valid/五日期 accuracy、macro-F1、每epoch训练历史、最优epoch、参数/时间、预测与checkpoint。

预定判定：token归因候选需 `residual_window−residual_constant` 的 valid macro-F1 均值>0、至少2/3 seed正，且五日期均值差至少4/5为正、其平均>0。部署候选另需 `residual_window−df_only` 的 valid均值≥+0.5pp、至少2/3 seed正，五日期至少4/5均值正且五日期差的平均≥+0.5pp。两关均过才称 DF 主模型正向候选；否则如实保留局部证据，不在同valid追调损失/结构。报告valid→Day270绝对降幅，但不将高绝对F1混称抗漂移。

停止：配置非frozen、CUDA不可用、已有最终预测或checkpoint、source/valid方向重复、输入/损失非有限、输入形状/类别异常、超预算则停并保留失败记录。原始数据只读，产物仅写入本run。
