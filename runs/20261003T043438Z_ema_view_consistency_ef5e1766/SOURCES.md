# 来源与复用

- `artifacts/prepared.pt`：复用 `20261003T030410Z_wide_bn_view_aligned_kd_43bb4cee` 已冻结的 TemporalDrift source/valid prepared 输入；不访问未来数据。
- `model.py`：显式复制该 run 的 wide progressive generator + BatchNorm + Transformer 结构，仅将条件名改为 CE/EMA 一致性条件。
- `framework_base.py` 及其他基础文件：从上述 run 显式复制，作为最小运行依赖。
- 本轮不加载历史教师 checkpoint、教师 logits 或固定教师模型；EMA 副本由每个学生从头训练时在线维护。
