# 来源与复用

- `artifacts/prepared.pt`：复用已冻结的 TemporalDrift source/valid 输入，来源 run `20261003T071914Z_peer_attribution_b4513513`；本 run 按自身 hash 记录。
- 模型基础代码显式复用当前 wide+BN 实现；未加载旧 checkpoint 初始化。
- 本轮唯一训练目标改动是预注册的监督对比损失及训练期投影头，推理不依赖投影头。
