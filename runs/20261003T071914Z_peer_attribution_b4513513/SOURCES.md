# 来源与复用

- `artifacts/prepared.pt`：复用 `20261003T053813Z_peer_symmetric_consistency_3057fd74` 的冻结 TemporalDrift source/valid prepared 输入。
- `model.py`、基础模块和输入处理：显式复用上一轮单网络双视图实验；本轮只改变损失中第二视图是否参与 CE。
- `artifacts/ce_only_*`、`dual_mask_ce_*`、`mask_peer_*`：复制上一轮已完成结果作为只读参考，不作初始化、不重训。
- 本轮不加载固定教师或 EMA；每个新条件只有一个可训练模型。
