# DF 主模型直接融合生成 window token：结果

状态：completed。冻结的三条件×三 seed×45 epochs 全部完成；RTX 4090 GPU0，耗时 145.08 秒（含输入生成、训练及评分）。独立 `verify.py` 复核通过，详情见 `artifacts/integrity.json`。

## 结论

预定的 DF 上 token 候选门槛明确失败。`df_window` valid macro-F1 三 seed 均值 35.358%，比等结构、等参数的 `df_constant` 46.364% 低 11.006 个百分点，三 seed 配对差均负；五个开发未来日期的均值分别低 6.409～8.441pp。它也明显低于原始 `df_only` 的 valid 45.681%。因此，上一轮轻量 CNN 的直接融合增益没有推广到当前 DF＋window MLP＋拼接头。不能以此宣称生成器普遍无效，只能否定本次冻结的 DF 集成方案。

## 分类结果

每格为 accuracy / macro-F1，单位 %，三 seed 均值。逐 seed 完整数值在 `artifacts/metrics.csv`，预测在 `artifacts/predictions.npz`。

| 角色 | DF-only | DF＋常量分支 | DF＋真实 window |
|---|---:|---:|---:|
| source | 89.935 / 89.842 | 86.193 / 85.774 | 79.935 / 79.574 |
| valid | 47.451 / 45.681 | 47.451 / 46.364 | 36.601 / 35.358 |
| Day14 | 43.382 / 42.324 | 43.088 / 42.081 | 34.526 / 33.640 |
| Day30 | 39.984 / 38.692 | 40.490 / 39.204 | 32.304 / 31.529 |
| Day90 | 33.873 / 31.803 | 34.183 / 32.542 | 26.438 / 25.134 |
| Day150 | 30.343 / 28.579 | 30.065 / 28.877 | 23.219 / 22.287 |
| Day270 | 27.925 / 25.583 | 28.448 / 26.299 | 21.013 / 19.890 |

`df_window − df_constant` 的 valid macro-F1 配对差（seed 1729/3407/2026）为 −12.045/−9.037/−11.936pp。Day14/30/90/150/270 的三 seed 均值差分别为 −8.441/−7.675/−7.408/−6.590/−6.409pp，五日期全部负。相对原始 DF，valid 差 −10.324pp，五日期为 −8.684/−7.163/−6.669/−6.292/−5.693pp。预定门槛要求 valid 均值正、至少2/3 seed正、未来至少4/5日期均值正；实际为零 seed、零日期。

最优 epoch：DF-only 42/44/44，常量分支 37/37/41，真实 window 40/18/21。第45轮训练 accuracy 均值约 73.105/72.990/93.088%，但所选模型 source macro-F1 约 89.842/85.774/79.574%；末轮训练态 accuracy 与所选 checkpoint 的 eval 态 source 指标不可直接等同。真实 window 较高的末轮训练拟合、较早的部分最优 epoch 和较低的 valid/未来分数共同提示过拟合或集成不匹配，不能据此单独确定根因。DF-only 3,718,374 参数；两个融合条件各 3,778,790 参数，后两者初始化、batch 顺序、预算完全匹配，DF 主干初始权重三条件一致。

valid→Day270 的 macro-F1 绝对下降为 DF-only 20.098pp、常量分支 20.065pp、真实 window 15.468pp。真实 window 起点和终点都低得多；较小下降不代表抗漂移更好。此前轻量 CNN 融合实验的 valid F1 33.348% 也低于本轮 DF-only 45.681%，显示骨干强度对比较很关键；两轮网络与集成结构不同，不能把差异归因于单一组件。

## 数据、实现与核验

复用固定 TemporalDrift source2040、valid510、五日期各2040行；source 标签训练，valid 标签仅选模，未来标签仅选模后作已观察开发评分。输入前5000包仅取方向，window 宽50/250、partial 中和、source-only统计标准化，与上一轮直接融合输入规则一致。DF 采用本项目 `src/ta_wf_next/models/df.py`，从头训练；没有加载旧 checkpoint、访问 WTT-Time/AWF 或修改原始数据。版本与依赖散列在 `artifacts/input_seal.json`，方案及预算在 `PLAN.md`、`config.json`。

独立核验：source/valid 方向重复0，source-only统计和输入 seal 相符，63 个 checkpoint 预测与63行指标重算、最早最佳 epoch 与初始化一致性检查均无错误。当前结果属于 TemporalDrift 方法开发，不是外部确认。若继续研究 DF 集成，应将这次阴性结果保留，先做预定机制诊断（例如分支尺度与优化交互），而不是在同一 valid 上反复调融合权重并只报告最优版本。
