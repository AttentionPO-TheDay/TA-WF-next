# 事后直接重排对照

此诊断在训练配方冻结后添加，使用相同已冻结的Top5、T2和源邻域投票公式；没有新训练、阈值搜索或目标返调。它不是预注册比较。所有3seed×10slice完整报告。

| 直接推理方法 | F1 % | ΔF1 pp | NAR % | Recovery % |
|---|---:|---:|---:|---:|
| direct_candidate_rerank | 42.679±0.041 | 0.799±0.088 | 4.807±0.256 | 5.377±0.100 |
| direct_source_knn | 42.282±0.036 | 0.402±0.109 | 9.469±0.640 | 8.250±0.190 |

direct_candidate_rerank直接取本轮soft_neighbor固定目标分布argmax；direct_source_knn直接取两bank源邻居平均投票argmax（不限制Top5）。用于检查适应是否超过简单的源证据重排，不据此重写本轮训练或冒充新确认。
