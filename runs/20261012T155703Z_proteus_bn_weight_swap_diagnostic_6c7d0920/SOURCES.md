# 来源

- 基础 checkpoint、prepared 数据和 split 只读复用既有审计 run。
- `full_strength025` checkpoint 来自已完成并审计的 `20261011T042912Z_proteus_strength_sweep800_b022cde2`。
- 新代码只做 state_dict 权重/BN running statistics 交换和固定 query 推理。
