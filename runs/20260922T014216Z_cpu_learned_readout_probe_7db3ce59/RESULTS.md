# 实验结果

## CPU learned readout 完成（修订版 v1）

固定抽样与上一实验一致：source 2040、valid 510、JP 2036、subpage 2040；抽样行号逐角色逐项相同。source 标签用于拟合，valid 只报告，JP/subpage 只作已暴露开发评分。使用 NumPy/scikit-learn 的 `SGDClassifier(log_loss, alpha=1e-4, max_iter=50, tol=1e-3, seed=1729)`，每个视角独立 StandardScaler；没有融合、超参扫描或 GPU。

| 视角 | source valid accuracy / macro-F1 | JP accuracy / macro-F1 | subpage accuracy / macro-F1 |
|---|---:|---:|---:|
| packet | 11.57% / 9.93% | 9.04% / 8.04% | 4.85% / 4.24% |
| windows | 20.20% / 19.02% | 16.94% / 16.40% | 8.19% / 7.68% |

## 解释

1. packet 的 learned 线性读出在 source valid 与上一轮原型相同（11.57%），但 JP 低 1.33 pp、subpage 低 1.67 pp。因此当前 packet 弱读出不能简单归因于“最近类原型太弱”；至少在这个固定 CPU 线性预算下，没有发现可利用的额外线性信号。
2. windows 与上一轮原型读出接近：valid -0.39 pp、JP +0.49 pp、subpage -0.05 pp。没有稳定的 learned 增益，也不足以证明稳健互补或抗漂移。
3. 两个视角都在 50 个 epoch 达到上限并发出未收敛警告；所以结果是固定有限预算 probe，不是充分优化的神经网络基线，也不能否定 token embedding 或深度模型。

## 完整性与限制

预测先封存后评分；修订版 `artifacts/predictions.npz` SHA-256 为 `d6e9e170d8dd66224e76c94fba4e9a8ae636127ed56593aad23fa75b909c26a0`。初始 v0 因误把 window 的 `observed_count/partial` 作为输入列而作废，工件保存在 `artifacts/invalid_v0/`；v1 恢复与上一实验完全相同的两列窗口特征后重跑。没有读取 WTT/AWF 或新条件。CPU probe 不能替代后续 GPU learned baseline；但它支持暂不设计 selector，并优先把 GPU 实验集中在 packet/window 的容量匹配深度基线和 run-token adapter 上。
