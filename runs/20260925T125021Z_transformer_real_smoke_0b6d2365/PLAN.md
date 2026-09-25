# 20260925T125021Z_transformer_real_smoke_0b6d2365

问题：多视角 Transformer 原型在固定 TemporalDrift source 四条真实 trace 上能否完成预训练、微调及 adapter-only TTA 单步闭环？

状态：frozen；仅 source 四条真实 trace 的结构 smoke，不评分。

数据：`configs/datasets.json` 的 TemporalDrift/train.npz，复用既有已观察 source 行号 [8800,15843,687,10601]；标签只给 finetune step，不读 valid/未来。输入经正式 `traffic_views.py` 和 `batch_from_views` 生成 packet/run/window；不创建新 split、不选择 epoch、不计算 F1。

动作：一轮 label-free span reconstruction、一步有标签 CE、一步冻结 teacher + adapter-only TTA。CPU，固定 seed 6238；只检查 shape、有限 loss、梯度和冻结隔离。任何 NaN、mask/预算错或 query 访问即停止。
