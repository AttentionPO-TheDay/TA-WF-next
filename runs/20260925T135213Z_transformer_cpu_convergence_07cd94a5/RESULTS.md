# 实验结果

状态：completed。TemporalDrift source/valid 收敛诊断完成；未来日期、WTT/AWF 和 TTA 均未访问。9 组条件×seed、18 组 train/valid 预测均已保存，独立核验 18/18、0 errors。

## 主要结果

固定上一轮输入、模型、5-shot 抽样和三 seed，仅将监督微调从 25 轮延长到 100 轮。以下为每个 seed 按 valid Macro-F1 选择的最优 checkpoint 均值（百分比）：

| 条件 | train accuracy | train Macro-F1 | valid accuracy | valid Macro-F1 | 最优 epoch 均值 |
|---|---:|---:|---:|---:|---:|
| packet scratch | 31.57 | 25.60 | 15.16 | 11.81 | 93.3 |
| 多视角 scratch | 40.07 | 34.54 | 15.69 | 12.91 | 98.3 |
| 多视角预训练 | 45.16 | 39.83 | 16.93 | 14.84 | 93.3 |

多视角预训练相对多视角 scratch 的 valid Macro-F1 提高 1.92 个百分点；三个 seed 的差值为 +4.27、+1.79、−0.29 个百分点，因此是弱正向信号而非稳定收益。相对旧 25 轮结果（packet 3.73%、多视角 scratch 2.63%、多视角预训练 4.51%），100 轮确实大幅改善，但训练预算和选模机会不同，不将其当作严格独立性能确认。

## 收敛解释

训练轮数不足是低分的重要原因：第 25 轮的三条件平均 train eval accuracy 只有约 8.8%、8.6%、10.4%，到最优 checkpoint 上升到 31.6%、40.1%、45.2%；valid F1 也从第 25 轮约 2.39%、1.84%、3.18%上升到 11.81%、12.91%、14.84%。

但模型仍未达到充分拟合：9 个 checkpoint 中有 4 个最优点在第 100 轮，预训练条件平均 train accuracy 仍低于 50%，而且 valid 仍明显低于既往同 5-shot 轻量 CNN 约 20.30% 的结果。因此不能把问题归结为训练轮数，也不能据此宣称 Transformer 或生成器已经可用。

预训练条件在验证集上预测的类别覆盖为 85/94/92 类，高于 packet scratch 的 73/78/83 类，说明同域重建可能改善了表示覆盖；但收益跨 seed 不完全稳定，且本轮没有未来日期评分，不能推出抗漂移效果。

## 结论与下一步

本轮支持“25 轮过少”这一判断；100 轮时尚未充分拟合的根因仍待区分，可能涉及 token 压缩/摘要表示、分类头或优化。下一步不应直接做 TTA 或继续无界延长训练；应先做预算匹配的表示消融（解除 packet 50 包聚合或 run 128 截断中的一个），并保留相同训练预算和 packet 对照。只有源域分类达到合理水平后，才有意义评价生成器对时间漂移的贡献。

## 核验与版本

`artifacts/integrity.json` 报告 `prediction_sets_checked=18`、`errors=[]`、模型参数 110400。固定输入 `prepared.pt` SHA-256 为 `4dc489bf088246c59b73cdb071bb22459c975051ed84b27962bbbf0067d70b24`；Transformer 原型 SHA-256 为 `ba921e57111694effd9df4b00e3da7adf3b538a931c047a15f64d1626fc868b6`。本轮仅使用 CPU，原始数据只读。
