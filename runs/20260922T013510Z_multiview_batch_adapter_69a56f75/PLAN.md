# 20260922T013510Z_multiview_batch_adapter_69a56f75

问题：实现多视角 learned baseline 所需的显式 batch、mask 与截断接口，并验证 coarse-only 无精确长度旁路。

状态：frozen implementation check。只修改本项目代码与合成单元测试；不读取真实数组或标签、不训练、不评分、不使用 GPU。

实现范围：packet direction、exact run、coarse run、固定宽度 direction window。每种 batch 显式返回 token mask、未截断 token 数、原始 observed packet 数和截断标记。observation budget 与 batch width 分开。

信息边界：coarse run 仅提供方向、log2 count bin、相邻 bin 差及其有效位、边界标记；不能包含 exact count。时间和 size 本轮不做 batch adapter，避免未经审计地把 None 转为零。

检查：合成输入形状、padding mask、截断、partial window、coarse/exact 分离和参数拒绝；随后运行全项目测试。零随机数、零训练、零 checkpoint、零 GPU。
