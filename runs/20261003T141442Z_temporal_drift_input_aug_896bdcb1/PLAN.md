# 20261003T141442Z_temporal_drift_input_aug_896bdcb1

## 问题

固定 wide+BN Transformer 已经使用既有 90-bin 随机遮挡，但该遮挡不改变时间原点。新增训练期输入增强 `temporal_origin_jitter`：对两路 TAM 时间计数同步做保持局部结构的全局时间平移，模拟网站流量整体到达时间的轻微漂移。它只作用于 source 的训练 batch；valid 始终使用原始输入。

## 条件

- `ce_control`：沿用既有训练流程，只使用预声明的 90-bin span mask。
- `temporal_origin_jitter`：先使用同样的 90-bin span mask，再以 0.75 概率将两路 TAM 同步平移 -2、-1、0、+1 或 +2 个时间 bin（每 bin 15 个原始位置），移出边界的计数丢弃，空出的边界补零；不重排方向、不改变总类别标签。
- 两条件共享模型、初始化、source index stream、span-mask stream、优化器、步数、验证选模规则和三 seed。
- 推理与 checkpoint 评价不应用 jitter。固定验证遮挡诊断仍单独使用原有 span mask。

## 数据与判定

TemporalDrift source 15300 / valid 510 / 102 类。source 标签只用于训练；valid 标签只用于预声明 checkpoint 选择和指标。未来日期、WTT-Time、AWF、TTA 均关闭。主比较为 `temporal_origin_jitter - ce_control`，要求平均 accuracy 至少 +1pp、3/3 seed 正、Macro-F1 不下降；否则 FAIL。

## 预算

GPU2，最多两路并行；每任务 7200 秒，总计 43200 秒；FP32、无 AMP/TF32、确定性。该 run 复用上一轮已封存的 prepared source/valid 输入，仅复制并冻结哈希，不读取未来数据。

## 启动前修正

尚未训练或评分前，修正复制审计脚本的历史路径：本轮参考 run 使用 ce_control 而非 ce_only。时间平移在移出边界时丢弃计数，不宣称总质量守恒。冻结哈希随此实现核对重新登记；增强参数与评价规则不变。GPU2 在沙箱外可访问，此前驱动不可通信属于沙箱访问限制。
