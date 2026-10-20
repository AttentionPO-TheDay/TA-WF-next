# 纯净源域双协议轻量超参数敏感性

问题：在已观察的 TemporalDrift Day14/90/270 上，`no_pseudo_fixed` 对学习率及终端步数、`tent_entropy_gate` 对学习率及熵区间有多敏感？用户明确选择两种配方都做。本轮是开发诊断，不是新的确认测试；选择这些轴受到此前已观察矩阵结果影响，不能把其外部 WTT/AWF 结果返调为本轮独立验证。

固定源底座为 `20261018T125337Z_pure_source_backbone_c1d249f6` 三 seed 纯净 wide_bn；同 seed 的 source official-valid checkpoint 已选定，不新增选模机会。沿用核心矩阵 `20261018T144952Z_pure_temporal_core_matrix_v2_4aa89f43` 的输入张量、1024条无标签 reference、独立 query、模型代码、Adam betas、batch128、source/support/dropout随机流。无伪标签配方仍为 source CE + target entropy/diversity + Gaussian MMD，全网络除分类头以外适应；门控 TENT 仍固定 BN 统计与分类头，只更新 Transformer 与末端 LayerNorm 仿射。query 标签只用于终端指标，query 输入不进入适应或超参数选择。

七种新配置各在 seed 42/100/2026 × Day14/90/270 全部9单元执行：无伪标签配方 `no_pseudo_lr_half_800`=0.000125/800步、`no_pseudo_lr_double_800`=0.0005/800步、`no_pseudo_steps_half_400`=0.00025/400步；门控 TENT `gate_lr_half_800`=0.000125/800步、`gate_lr_double_800`=0.0005/800步、`gate_narrow_800`=0.00025/800步且熵区间[0.25,0.35] nat、`gate_wide_800`=0.00025/800步且区间[0.1,0.5] nat。控制为核心矩阵已审计的 `no_pseudo_fixed` 与 `tent_entropy_gate`（均lr=0.00025/800步；TENT门控[0.2,0.4] nat），各九单元只读复用，不当作本轮重跑。预先固定全部七种单因素改动，不依据目标标签挑选后续运行次序、最佳 seed 或终端步；不做多参数组合搜索。

主输出是每配置九个配对 seed×日期的固定102类 Macro-F1 相对**本配方控制**差（百分点），同时报告逐日期三seed均值与样本SD、Accuracy、NAR、Recovery、正向单元数、时间和显存；三日期等权，不把九单元当九个独立数据集或做显著性推断。NAR分母为同 seed×日期冻结模型原正确数，Recovery为原错误数。记录训练成本和400/800步支持集 logit L2/熵轨迹，不用轨迹挑 checkpoint。阴性、崩溃及全部单元都报告；结果为局部敏感性，不证明全局最优或某组件因果贡献。

执行门槛：检查原核心矩阵的45条独立重载审计、所有源权重/目标切分/代码SHA、source-valid-reference-query输入隔离；在仅源输入上运行两步梯度及作用域 smoke，确保分类头与BN buffer分别按配方锁定，派生实现与原版除指定超参数外一致，然后冻结本 run。正式执行拒绝 draft。三个 seed 在空闲 GPU0 上单进程顺序执行，每单元≤2400秒，总墙钟≤21600秒、输出≤2GiB；任何非有限数值、散列或隔离断言失败则停止并保留部分产物。结束时重载63个新checkpoint、全query复算81条指标（含18历史控制）才标 completed。
