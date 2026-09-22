# 实验结果

## 完成状态与冻结选择

Historical Retention Baselines on Frozen Global Representations 已按 `HISTORY_RETENTION_PLAN.md` 完成。新增 backbone 训练/微调、checkpoint 写入均为 0；只读复用了 `exp_b471517a3e6f41e7` 的 18 个固定 manifest、36 个 donor 评价 artifact、两套 G-source，以及 `exp_9121b664a1854097` 的 DF epoch 29/VarCNNDirection epoch 23 checkpoint。未重抽 support、未改变 query/shot/seed、未重做隔离审计、未覆盖 donor。

preflight 完整解析并核对了 20.7 MB manifest、144 行 donor 指标、36 个 donor evaluation 哈希、split、隔离审计、两个 joblib 和 checkpoint/history，错误为 0。source-only 共享选择使用 v3 source-holdout 的嵌套 3/10-shot 伪 support 与 official source validation；没有访问 current query。三个 alpha 的 12 单元 mean macro-F1 为 69.994/68.704/66.613%，选择 `alpha=0.25`；三个 lambda 为 72.437/72.670/72.614%，选择 `lambda=0.1`。36 次 source-side shrinkage 均收敛，最多 6 次迭代。

## 绝对性能

单元格为三个 support seed 的 `accuracy±SD / macro-F1±SD`，均为百分数。Current prototype 是 donor `simple_maintenance` 的同一定义与预测。

### DF

| 日期 | shot | G-source | Current prototype | G-current | Prototype interpolation | Shrink-to-source |
|---|---:|---:|---:|---:|---:|---:|
| Day14 | 3 | 72.50±0.05 / 72.34±0.05 | 64.50±1.53 / 63.87±1.50 | 64.65±1.48 / 64.04±1.50 | 70.75±0.14 / 70.51±0.14 | 72.48±0.11 / 72.32±0.12 |
| Day14 | 10 | 72.56±0.07 / 72.39±0.07 | 70.12±0.12 / 69.81±0.15 | 70.67±0.13 / 70.44±0.16 | 71.13±0.09 / 70.92±0.07 | 72.56±0.06 / 72.41±0.07 |
| Day90 | 3 | 57.27±0.01 / 55.44±0.00 | 55.26±0.64 / 54.14±0.87 | 55.76±0.59 / 54.65±0.75 | 57.33±0.14 / 55.83±0.12 | 57.28±0.02 / 55.52±0.01 |
| Day90 | 10 | 57.28±0.03 / 55.45±0.02 | 62.25±0.16 / 61.54±0.14 | 63.72±0.24 / 63.25±0.21 | 57.57±0.16 / 56.09±0.15 | 57.31±0.03 / 55.55±0.02 |
| Day270 | 3 | 46.70±0.01 / 43.90±0.01 | 49.02±0.93 / 47.28±0.88 | 49.50±1.08 / 47.89±0.97 | 48.54±0.20 / 45.92±0.25 | 46.83±0.03 / 44.07±0.03 |
| Day270 | 10 | 46.71±0.01 / 43.90±0.01 | 57.25±0.01 / 55.99±0.26 | 58.97±0.34 / 58.15±0.45 | 48.99±0.10 / 46.42±0.10 | 46.78±0.02 / 44.02±0.01 |

### VarCNNDirection

| 日期 | shot | G-source | Current prototype | G-current | Prototype interpolation | Shrink-to-source |
|---|---:|---:|---:|---:|---:|---:|
| Day14 | 3 | 70.85±0.06 / 70.70±0.06 | 55.19±2.76 / 54.43±2.89 | 55.04±2.69 / 54.31±2.76 | 68.42±0.09 / 68.07±0.06 | 70.85±0.06 / 70.70±0.06 |
| Day14 | 10 | 70.89±0.05 / 70.75±0.05 | 65.74±0.74 / 65.28±0.75 | 66.25±0.53 / 65.95±0.58 | 69.04±0.11 / 68.73±0.10 | 70.92±0.04 / 70.79±0.03 |
| Day90 | 3 | 57.31±0.00 / 55.46±0.00 | 51.29±0.22 / 50.39±0.27 | 52.08±0.35 / 51.16±0.39 | 56.62±0.21 / 55.19±0.19 | 57.30±0.04 / 55.52±0.05 |
| Day90 | 10 | 57.32±0.07 / 55.47±0.05 | 61.43±0.51 / 60.87±0.62 | 63.21±0.34 / 62.79±0.42 | 57.10±0.12 / 55.69±0.15 | 57.32±0.05 / 55.52±0.04 |
| Day270 | 3 | 47.58±0.05 / 44.71±0.05 | 47.86±1.12 / 46.61±1.08 | 48.39±1.09 / 47.14±1.07 | 49.19±0.17 / 46.35±0.16 | 47.70±0.04 / 44.87±0.02 |
| Day270 | 10 | 47.52±0.07 / 44.64±0.06 | 59.12±0.53 / 58.20±0.63 | 60.96±0.33 / 60.44±0.30 | 49.84±0.04 / 47.10±0.07 | 47.66±0.08 / 44.83±0.06 |

## Day14 保护与 Day90/Day270 恢复

两种简单历史保留都未通过预注册的“同时足够”门槛。

- Interpolation 在 Day14 明显优于 current-only，但仍比 G-source 低 1.48–2.63 pp macro-F1，因此两个 backbone 都未通过 Day14 保护门槛。对有正 current-only 恢复的后期单元，它只保留 DF Day90-10/Day270-3/Day270-10 的 8.17%/50.61%/17.65%，以及 VarCNNDirection 的 3.05%/67.55%/15.57%，远低于 80%。
- Shrinkage 基本复现 G-source：Day14 相对 G-source 为 DF -0.01/+0.01 pp、VarCNNDirection +0.01/+0.04 pp（3/10-shot）。它在 DF 10-shot 相对较强 current-only 的提升为 1.96 pp，距预注册 2.0 pp 门槛差 0.04 pp，因此严格判断下没有跨 backbone 通过 Day14 门槛。更重要的是后期保留率仅为 DF 1.26%/4.40%/0.88% 与 VarCNNDirection 0.75%/6.56%/1.16%（依次 Day90-10、Day270-3、Day270-10）。
- Day90-3 的 current-only recovery 在两模型均为负，按计划不以负分母计算有利保留率。

这表明固定 source-side 规则给出的两种常规基线位于明显的保守端：越接近 G-source，Day14 越安全，但越无法利用 10-shot 在 Day90/270 可提供的大幅恢复。它们没有同时解释“早期受损、后期恢复”。

## 相对 G-source 的双向错误转移

下表为三个 seed 的 `corrected/harmed/net` 平均计数；每个 seed 的原始计数、比例与 SD 见 `error_transfers.csv`，102 类拆分见 `per_website_transfers.csv`。

| Backbone | 日期 | shot | Interpolation | Shrinkage |
|---|---|---:|---:|---:|
| DF | Day14 | 3 | 689.3 / 1079.7 / -390.3 | 43.3 / 48.0 / -4.7 |
| DF | Day14 | 10 | 651.3 / 958.0 / -306.7 | 28.0 / 26.3 / +1.7 |
| DF | Day90 | 3 | 1089.7 / 1073.0 / +16.7 | 85.0 / 83.0 / +2.0 |
| DF | Day90 | 10 | 1028.3 / 949.3 / +79.0 | 61.0 / 54.0 / +7.0 |
| DF | Day270 | 3 | 1008.0 / 648.0 / +360.0 | 62.0 / 37.7 / +24.3 |
| DF | Day270 | 10 | 993.7 / 562.7 / +431.0 | 42.0 / 29.3 / +12.7 |
| VarCNNDirection | Day14 | 3 | 920.3 / 1460.0 / -539.7 | 63.0 / 62.3 / +0.7 |
| VarCNNDirection | Day14 | 10 | 863.0 / 1263.3 / -400.3 | 38.3 / 31.0 / +7.3 |
| VarCNNDirection | Day90 | 3 | 1377.0 / 1570.7 / -193.7 | 76.0 / 78.7 / -2.7 |
| VarCNNDirection | Day90 | 10 | 1297.3 / 1358.3 / -61.0 | 56.3 / 57.0 / -0.7 |
| VarCNNDirection | Day270 | 3 | 1194.7 / 879.3 / +315.3 | 69.7 / 47.3 / +22.3 |
| VarCNNDirection | Day270 | 10 | 1180.3 / 742.0 / +438.3 | 63.0 / 36.0 / +27.0 |

跨全部 36 个配置（同一 trace 在不同 seed/shot 出现时按不同 sample-evaluation 计），interpolation corrected 36,879、harmed 37,633、net -754；shrinkage corrected 2,063、harmed 1,772、net +291。净计数不能替代 macro-F1，因为网站支持和错误分布不同。

## 网站集中与跨 backbone 重复性

按预注册规则（同一日期合并两 shot，至少 2/3 seeds 同号；两个 backbone 各至少 2/3 dates 重复）：interpolation 有 28 个重复净受益网站、34 个重复净受损网站；shrinkage 为 16/13。完整 ID、每模型合格日期数和聚合计数见 `posthoc_repeatability.json`。

- Interpolation 最大聚合净受益为网站 69/40/27：+781/+675/+580；最大净受损为 83/66/58：-1184/-1069/-1002。
- Shrinkage 最大聚合净受益为网站 92/40/44：+92/+50/+43；最大净受损为 31/70/66：-44/-42/-40。

因此收益与损害确实集中在固定网站，并非只由单一 seed 或单一 backbone 驱动；但方向并不只表现为可利用的统一“坏网站”集合，不同保留方法的重复受损集合也明显不同。

## 3-shot 瓶颈

10-shot−3-shot macro-F1 gap 在 DF 的 G-current 为 Day14/90/270 +6.40/+8.60/+10.27 pp，在 VarCNNDirection 为 +11.64/+11.63/+13.30 pp；current prototype 同样为 +5.94/+7.40/+8.71 和 +10.85/+10.48/+11.59 pp。

Interpolation 的 gap 仅为 DF +0.41/+0.26/+0.50 pp、VarCNNDirection +0.67/+0.50/+0.75 pp；shrinkage 为 DF +0.08/+0.03/-0.05 pp、VarCNNDirection +0.08/+0.00/-0.04 pp。按预注册的“gap 缩小至少 50%且 3-shot 不低于 G-source 超过 1 pp”的数值门槛，interpolation 在每个 backbone 的 Day90/270 通过，shrinkage 在全部日期通过。

但这不是有意义的低标签恢复：gap 主要因两种历史方法同时压低 10-shot 适应量而消失。3-shot 的后期改善仍很小，而 10-shot current-only 的 7–16 pp recovery 基本丢失。因此应区分“shot gap 的形式门槛已缩小”和“3-shot 已能提供后期恢复”；后者仍未解决。这个解释不改变预注册门槛结果，也不把 under-adaptation 包装成标签效率改善。

## 可观测失效

固定可见量能识别“靠近边界、预测容易改变”的样本，但不能可靠区分改动会修复还是损害。以 harmed 为正类、corrected 为负类的逐配置 AUC：

- Interpolation：G-source margin 在 DF/VarCNNDirection 均值 0.567/0.583；interpolation margin 为 0.453/0.429；最大 prototype cosine 为 0.482/0.471。
- Shrinkage：G-source margin 为 0.483/0.497；shrinkage margin 为 0.462/0.458。

这些值大多接近 0.5，且各配置范围有重叠。重复受损网站的 source-current prototype cosine 并不低：interpolation 为 DF 0.903、VarCNNDirection 0.891（其他网站 0.890/0.887）；shrinkage 为 0.925/0.918（其他 0.889/0.884）。本实验因此发现跨模型重复的网站级得失，但没有发现一个已足够区分 harmful 与 corrective update 的简单预测时可见指标。不能据此直接开启或设计新机制。

## 五个问题的直接回答

1. **简单历史保留是否已经足够？** 否。两方法均未同时通过跨 backbone Day14 保护和后期恢复保留门槛；shrinkage 保护早期但几乎不适应，interpolation 有少量后期收益但仍牺牲大量 10-shot recovery。
2. **修复/损害多少？** Interpolation 聚合为 36,879 corrected、37,633 harmed、net -754；shrinkage 为 2,063/1,772/+291。逐配置均值/SD 与计数见正式 CSV 和上表。
3. **是否集中在固定网站且跨 backbone 重复？** 是。Interpolation 有 28 个重复净受益、34 个重复净受损网站；shrinkage 为 16/13，并有显著的 top-site 集中。
4. **3-shot 瓶颈是否缓解？** 形式上的 3/10-shot gap 明显缩小并通过预注册门槛，但原因是压制了 10-shot recovery；真正的 3-shot 后期恢复瓶颈仍未解决。
5. **还剩哪些跨模型重复且可观测失效？** 固定网站的收益/损害跨模型重复；低 margin 可观测到“易变”，却不能可靠预判改动是修复还是损害，prototype alignment 也不分离重复受损网站。因此目前没有得到一个足够明确的可观测失效规则。

## 完整性、产物与限制

- `metrics_long.csv` 216 行；`error_transfers.csv` 180 行；`per_website_transfers.csv` 18,360 行；36 个评价 JSON 保存 4,978,656 条 method prediction、逐网站指标、错误转移和可见诊断。
- 独立 verifier 重算全部 accuracy/macro-F1、每配置及每网站 corrected/harmed/net，并逐项比对 donor 预测、manifest、truth、选择权限和冻结哈希；`integrity_check.json` 为 `passed=true`、0 errors、0 nonconverged fit。
- `posthoc_repeatability.json` 保存正/负重复网站、top 集中和 harmed-vs-corrected AUC；该分析未反馈到 alpha/lambda 或预测。
- 固有限制：source-side 选择倾向 source retention，所考察的是预注册的固定小候选规则，不证明所有可能的 interpolation/shrinkage 都无效；TemporalDrift query 已是开发数据，不能当独立确认；shot-gap 门槛可被 under-adaptation 满足，故必须结合绝对 recovery 解释。

本实验到此停止，不扩展候选、模型、日期或复杂机制。是否另立实验由 Host 决定。
