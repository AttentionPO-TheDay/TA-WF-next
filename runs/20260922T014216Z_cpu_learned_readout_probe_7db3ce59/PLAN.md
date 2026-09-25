# 20260922T014216Z_cpu_learned_readout_probe_7db3ce59

问题：固定 CPU 监督线性读出能否显著改善 packet、run 与 window 视角的 source 辨别力，并复核已暴露 JP/subpage 开发表现。

状态：frozen_cpu_probe。使用与前一 token-effect 诊断相同的 seed=1729、每类 source 20/valid 5/JP 20/subpage 20 抽样。source 标签只用于监督拟合；valid 只作报告，不调参；JP/subpage 只作已暴露开发评分。无新条件、无 WTT/AWF。

对照：packet direction 5000 维；direction windows 50/250 的比例与转向率。每个视角独立 StandardScaler + 固定 `SGDClassifier(loss=log_loss, alpha=1e-4, max_iter=50, tol=1e-3, random_state=1729)`，不扫描超参、不融合、不路由。目标是判别读出能力，不是最终模型。

预算：单进程 CPU；显式 `CUDA_VISIBLE_DEVICES=''`；不加载 torch/checkpoint。预测先封存后评分。停止条件：数组结构、标签完整性或视角维度不符则失败。
