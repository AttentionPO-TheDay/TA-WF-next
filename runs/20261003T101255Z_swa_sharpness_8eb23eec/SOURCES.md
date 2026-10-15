# 来源与复用

- `artifacts/prepared.pt`：复用 `20261003T071914Z_peer_attribution_b4513513` 冻结的 TemporalDrift source/valid 输入；本 run 复制后按自身 hash 冻结。
- 模型和基础模块：显式复用上一轮 wide+BN 单网络代码；本轮只增加预注册后段 state 平均，不加载旧 checkpoint 初始化。
- 历史 CE 结果仅作为只读数值参考；SWA 由本轮三条新 CE 轨迹生成。
- 未加载教师、教师 logits、EMA、peer 目标或未来数据。
