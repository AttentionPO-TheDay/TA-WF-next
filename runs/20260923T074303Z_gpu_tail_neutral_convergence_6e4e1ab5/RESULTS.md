# 实验结果

状态：completed。GPU 0（RTX 4090）训练；source 2040、valid 510、各未来日期 2040，复用既有 TemporalDrift 抽样清单。full/tail-neutral 各 3 seed、45 epochs，同容量 240→128→102 MLP、同一 full-source 标准化。运行约 45.3 秒，未来日期只作开发评分。

## 主结果

Macro-F1，单位百分比；三训练 seed 平均。差值为 tail-neutral 减 full，单位百分点。

| 数据角色 | full | tail-neutral | 差值 | 三 seed 配对差 |
|---|---:|---:|---:|---|
| valid | 29.192 | 29.516 | +0.324 | +0.682 / -0.281 / +0.572 |
| Day14 | 26.827 | 27.936 | +1.109 | +0.209 / +0.742 / +2.376 |
| Day30 | 25.336 | 25.961 | +0.625 | +0.991 / -0.116 / +0.999 |
| Day90 | 21.207 | 21.174 | -0.033 | -0.735 / +0.253 / +0.383 |
| Day150 | 18.830 | 19.438 | +0.608 | +0.763 / +0.231 / +0.831 |
| Day270 | 16.597 | 17.005 | +0.408 | -0.298 / +0.701 / +0.821 |

tail-neutral 满足事先设定的候选门槛：valid 平均差正且 2/3 seed 正；未来日期 4/5 均值正，五日期均值差约 +0.543pp。但幅度小且 Day90 略负。valid→Day270 的绝对 F1 下降为 full 12.595pp、tail-neutral 12.511pp，仅缩小约 0.084pp；虽满足预设的方向性门槛，这个量级不能支持实质抗漂移结论。Day270 的绝对分数增加约 0.408pp，两 seed 正、一 seed 负。

## 收敛与限制

full 最佳 epoch 为 37/42/26；tail-neutral 为 39/44/41。第 45 轮训练准确率约 73–75%，valid 当轮 F1 均低于各自 best，说明继续增加 epoch 不会自动提升所选模型。相对前轮 15 epochs 的提高同时包含更长训练、更多 valid 选模机会和 GPU 数值路径差异，不视为独立重复或单独的算法收益。

这是对已经多次用于开发的 TemporalDrift 再评分；seed 不代表独立数据集。tail-neutral 是候选输入处理，尚未证明它能增强用户所关心的生成器训练价值，也未验证漂移后适配器更新。要评价主模型训练价值，仍需在容量、输入权限和预算匹配的 packet 主模型中比较生成 token 专属处理层开/关，并在冻结方法后做外部数据集评价。WTT-Time/AWF 本轮未访问。

## 核验

`artifacts/integrity.json`：42 组预测、42 行指标、42 次 checkpoint 重载重算，0 错误；source 标准化重算相等，valid 最高 F1 且并列最早 epoch 核对通过。输入、配置和代码散列保存在 `artifacts/pretraining_seal.json`，预测散列和抽样清单散列在 `artifacts/manifest.json`。
