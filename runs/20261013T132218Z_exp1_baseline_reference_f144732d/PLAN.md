# 20261013T132218Z_exp1_baseline_reference_f144732d

Exp-1 基线锁定只登记，不重复训练。它复用 MVI run 中已完成的 `no_pseudo_fixed`：固定学习率 0.00025、完全无伪标签、同一 TemporalDrift support/query split、3 seed×Day14/90/270。

该结果用于 Exp-2～Exp-4 的配对参考，所有新机制保持相同数据、标签权限和800次候选更新预算。
