# 来源与复用

- `artifacts/prepared.pt`：复用 `20261003T071914Z_peer_attribution_b4513513` 的冻结 TemporalDrift source/valid 输入。
- 模型和基础模块：显式复用上一轮 wide+BN 单网络代码；本轮只替换优化器步骤为两次前向控制与 SAM 扰动。
- `artifacts/ce_only_*`、对应 checkpoint：复制上一轮已完成的 CE 结果作为只读参考，不作初始化、不重训。
- 本轮不加载固定教师、教师 logits、EMA 或 peer 目标。
