# exp_3ae08f5b65a44cbf

Replication of Support-Internal Maintenance Across Backbone Training Seeds。冻结复用既有 10-shot support-internal historical-retention baseline，在 TemporalDrift 上以原 seed 6238 加两个预注册 backbone training seeds 1013/2024，分别检验 DF 与 VarCNNDirection 的训练随机性重复性；3-shot 仅作既有低标签 selection-noise 诊断。

状态：frozen。完整训练、评价、方差分解、门槛、信息权限与停止规则见 `REPLICATION_PLAN.md`。WTT-Time、AWF 与其他封闭最终评价均不可访问。
