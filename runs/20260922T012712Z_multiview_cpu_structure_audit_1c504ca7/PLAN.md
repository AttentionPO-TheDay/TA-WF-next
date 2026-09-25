# 20260922T012712Z_multiview_cpu_structure_audit_1c504ca7

问题：审计多视角表示的有效长度、run截断覆盖与位置对齐风险，为 CPU learned baseline 冻结输入规则。

状态：frozen_cpu_audit。只读加载 Proteus 数组的 X；不读取 y，不训练、不评分、不使用 GPU，不访问 WTT/AWF。

数据：`configs/datasets.json` 中的 `TemporalDrift`；train、valid、JP、BehaviorDrift/subpage。JP/subpage 已在前一实验中暴露，结果仅作开发诊断。每行最多检查前 5000 个观测；0 视为 padding，并审计内部 0。

指标：有效长度、内部 padding、方向 run 数/长度、各 token budget 的截断比例、窗口 partial 比例，以及跨数据角色的结构分布。脚本不读取标签，不按类别抽样。

预算与复现：单进程 CPU、NumPy、无随机数；只写本 run/artifacts。停止条件为数组缺失、非有限值或违反 padding 约束时失败；不根据结果修改表示规则。
