# 来源与复用

- `artifacts/prepared.pt`：复用 `20261003T043438Z_ema_view_consistency_ef5e1766` 的冻结 TemporalDrift source/valid prepared 输入；不访问未来数据。
- `model.py`：显式复制此前 wide+BN 模型结构；本轮损失只在同一模型的两个输入视图前向之间计算，不构造教师或 EMA 网络。
- `framework_base.py` 及其他基础文件：从上一轮 run 显式复制，作为最小运行依赖。
- 不加载任何历史教师 checkpoint、教师 logits 或 EMA 参数。

历史ce_only报告、预测、logits、初始化、索引及best checkpoint从上一轮复制到artifacts/ce_only_{seed}和checkpoints；checkpoint仅评价诊断，不作训练初始权重。prepared保持原hash。
