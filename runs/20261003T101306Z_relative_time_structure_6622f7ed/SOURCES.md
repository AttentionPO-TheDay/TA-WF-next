# 来源与复用

- `artifacts/prepared.pt`：复用 `20261003T071914Z_peer_attribution_b4513513` 冻结的 TemporalDrift source/valid 输入；本 run 复制后按自身 hash 冻结。
- 模型和基础模块：显式复用上一轮 wide+BN 单网络代码；本轮新增相对时间统计投影模块，未加载旧 checkpoint 初始化。
- 时间特征只使用当前输入的 signed timestamps，四个聚合统计和固定 120-bin 规则见 PLAN/config；不使用日期、标签或未来 query。
- 未加载教师、教师 logits、EMA、peer 目标或未来数据。
