# Proteus 机制忠实移植 baseline v1

## 目标与边界

本实验复现 Proteus 论文和固定提交 `Xinhao-Deng/Adaptive-WF-Attack@4cdab4163bf3de7036a2498dac533c888b97664d` 中的核心适应机制：源域监督回放、动态带宽 Gaussian-kernel MMD、目标熵最小化与 batch marginal entropy 多样性约束、基于预测熵的两分量 GMM 可靠性筛选，以及伪标签交叉熵微调。

官方发布脚本使用 RF、缺失的部分 TAM 文件、同一目标池进行适应和评估，并每个 epoch 用目标真标签记录最佳 F1；这些条件不符合本项目的独立 query 协议，不能直接作为严格公平结果。因此本 run 的名称和结论限定为：**Proteus 机制忠实移植到当前 wide+BN 底座，在 TemporalDrift 独立 support/query 协议下的 baseline**。它不是论文完整 RF 端到端复现。

## 模型与条件

固定当前项目的 wide+BN Transformer 三个 source-valid 最优 checkpoint（seed 21729、23407、22026），不重新训练 backbone。Proteus 适应阶段解冻整个模型，包含生成器、Transformer、分类器和 BatchNorm 参数；optimizer 为 Adam，lr=1e-3，与官方 `exp/proteus.py` 一致。

适应总计 100 个 epoch；每个 epoch 使用 1024 条 target support（batch=128，shuffle、drop-last）和 source train（复用冻结 checkpoint 对应的 source split，batch=128，shuffle、drop-last）循环。每 5 个 epoch 在 support 上重新计算 GMM 可靠性，并重建伪标签 loader。伪标签阈值为 0.6。

每个适应 batch 的总损失严格按官方实现相加：

`source CE + pseudo-label CE + target entropy + marginal entropy term + dynamic-bandwidth MMD`。

其中 target entropy 为样本熵均值，marginal term 为 `sum(mean_softmax * log(mean_softmax + 1e-5))`；MMD 使用官方代码的有偏核矩阵形式，带宽按当前 source/target feature 合并样本的全局平方距离均值计算并截断为至少 `1e-5`。GMM 使用 sklearn 两分量、`tol=1e-6`，选择均值较小的分量作为低熵可靠分量；归一化熵、GMM posterior、伪标签和所有更新均不读取 support 标签。

## 对照与评价

本 run 写入并核对以下对照：

1. `frozen`：复用前一轮冻结预测。
2. `global_entropy_adapter`：复用前一轮 10 步全局熵适应结果，作为轻量 TTA 参照。
3. `stepwise_rollback`：复用前一轮逐步接受与回退结果，作为当前候选机制参照。
4. `proteus_port`：本轮 100 epoch 完整机制移植。

Proteus 结果只在适应结束后对独立 query 运行一次；不按 query 标签选 epoch、阈值、GMM 分量、日期或 checkpoint。为检查训练是否崩溃，允许保存 support 无标签 loss、熵、GMM 可靠样本数和参数变化轨迹，但这些记录不使用标签。

主指标为独立 query 的固定 102 类 Macro-F1，辅指标为 Accuracy、负适应率、纠错率、query 改变率、support 伪标签覆盖率、各损失项和参数变化。冻结、global 和 rollback 的历史预测只读复用，不作为本轮新重复。

## 数据和权限

使用 TemporalDrift Day14、Day90、Day270；每个日期严格复用 `20261009T070330Z_tta_rollback_acceptance_91f808d9` 的 1024 条无标签 support 和独立 query。运行时逐行断言 split 一致。source train/valid 只用于复用当前 checkpoint 的训练角色和源 CE 回放；source 标签仅进入 source CE，与官方 Proteus 监督回放一致。support 标签完全不可用；query 标签只在最终预测固定后评分。

本轮不访问 WTT-Time、AWF、VersionDrift、NetworkDrift、BehaviorDrift 或外部未来测试。多因素评价需在本轮机制版本冻结后另立 run。

## 预算与停止条件

单 GPU、最多 3 小时/任务、9 个 Proteus 适应任务，共 900 个 epoch。出现 source/query split 不一致、非有限 loss、GMM 熵范围为零、伪标签少于一个 batch、模型参数未更新、query/support 交叉、冻结输入 hash 变化、CUDA OOM 或超时立即失败并保留产物。global 与 rollback 历史预测若无法逐行复现则停止，不将其结果混入汇总。

## 解释边界

若 `proteus_port` 优于 frozen，只能说明完整 Proteus 机制在当前 wide+BN 与独立 query 协议下有效；若不优于 global/rollback，不能据此否定论文 RF 结果，因为底座、输入、适应池大小、日期序列和官方 target-label 选模行为均不同。论文创新结论必须等待多因素开发评价和冻结后的外部评价。
