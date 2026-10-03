# 来源与复用说明

- `framework_base.py`、`base_model.py`、`encoder_base.py` 等为本项目上一轮 `20261002T053706Z_progressive_capacity_bn_mixup_1ad233da` 的显式最小复制，用于保持 wide+BN 生成器与 Transformer 接口一致。
- `wide_bn_base.py` 是上一轮 run-local `model.py` 的副本，仅用于初始化和输出接口核对。
- 本轮新增 `model.py` 的全局 TAM 统计量、统计投影和固定/门控残差；没有导入 `/home/rbf/TA-WF` 旧训练代码。
- baseline checkpoint、训练索引、输入缓存和标签来自上一轮已完成 wide+BN 实验，均作为历史复用并单列审计，不视为本轮重新训练。
