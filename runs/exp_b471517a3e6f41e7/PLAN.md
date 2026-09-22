# exp_b471517a3e6f41e7

Frozen Global Representation Temporal Recoverability Diagnostic。复用 `exp_9121b664a1854097` 的正式 DF 与 VarCNNDirection best checkpoint，在不训练或更新 backbone 的条件下，诊断最终 global embedding 在少量当前监督下的跨时间可恢复性。完整冻结定义见 `RECOVERABILITY_PLAN.md`。

状态：frozen；仅允许冻结特征抽取、source 线性读出、当前 support 线性读出、当前 prototype 读出及预定评分。新增 backbone 训练预算为 0。
