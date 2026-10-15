# 来源

- 模型、数据helper、runner骨架显式复制本项目 runs/20261014T070454Z_tent_baseline_v3_83d48bf3；本目录代码独立，不导入归档项目或其训练代码。来源链见该目录 SOURCES.md。
- official_tent.py及纯熵forward_and_adapt：DequanWang/tent，commit e9e926a668d85244c66a6d5c006efbd2b82e83e8；保留MIT许可。
- enhanced_tent.py是本次增强：固定源BN缓冲区，允许BN和指定LN affine梯度；不是官方TENT的原配置。优化函数逐字复用official_tent.py中的forward_and_adapt。
- 旧no_pseudo_fixed、Frozen及BN-TENT/批统计控制仅历史参考；来源、预测与hash记录在config/freeze。不重新计作重复。
