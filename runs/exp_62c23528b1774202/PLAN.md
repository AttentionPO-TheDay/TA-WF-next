# exp_62c23528b1774202

Equal-Budget Region-Level Historical Relevance Test。只检验：在 TemporalDrift、冻结 DF/VarCNNDirection、每类正式 3-shot 与相同历史存储预算下，区域级历史保留和冲突抑制是否稳定优于现有最强 support-CV 3-shot、普通多原型与网站级整体加权。

状态：frozen。不可变方法定义、信息防火墙、预算与裁决门槛见 `METHOD_PLAN_v1.md`。新增 backbone 训练/微调预算为 0；10-shot 只引用既有高标签预算背景，不进入本实验算法、选模或公平胜负。
