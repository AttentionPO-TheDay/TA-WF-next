# 实现来源

`model.py` 及其显式本地依赖 `base_model.py`、`candidate_base.py`、`encoder_base.py`、`framework_base.py`、`generator_base.py`、`input_base.py`、`rf_native.py`、`view_base.py` 复制自本项目已完成的 `runs/20261002T053706Z_progressive_capacity_bn_mixup_1ad233da`，不导入归档训练代码或 checkpoint。当前 `train.py` 按该 run 的 wide_bn 训练机制、AdamW、span mask、固定验证机会和最早最大 valid accuracy 规则重新实现单条件训练；源域行清单来自 `runs/20261009T014524Z_joint_role_identity_audit_51b4162b/artifacts/conservative_train_candidates.csv`。输入源由 `configs/datasets.json` 定位。

与旧训练不同之处：本 run 采用全部 18,553 条保守训练候选、完整官方 valid 2,160 行及 seed 42/100/2026；旧 run 为每类 150 条 source、每类 5 条 valid，seed 21729/23407/22026。旧性能不能作为本 run 的同配置结果。
