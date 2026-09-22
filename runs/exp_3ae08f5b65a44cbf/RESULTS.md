# Replication results

## 完成状态与严格结论

实验已按训练前冻结的 `REPLICATION_PLAN.md` 完成。新增训练严格为 4 次：DF×seeds 1013/2024、VarCNNDirection×seeds 1013/2024；没有补训第三 seed、延长 epoch、调超参或丢弃结果。原 seed 6238 未重训，直接复用 `exp_376fca9354214097` 的 36 个正式 common-query evaluation 并校验全部哈希。WTT-Time、AWF 和其他封闭最终评价未访问。

预注册的严格 replication **未通过**：18 个 architecture×training-seed×date 的 10-shot 主单元中 15 个通过。两个 architecture、三个 training seeds 的 **全部 12 个 Day90/270 单元均保留至少 80% current-only recovery**；VarCNNDirection 的三个 seed 也全部通过 Day14 保护。三个失败恰好是 DF 的三个 training seeds 的 Day14，selected 相对较强 current-only 仅高 1.951/1.701/1.313 pp macro-F1，均低于冻结的 2.0 pp 门槛。因此结果不是“原 seed 偶然失败、换 seed 后通过”，也不能把边界改写为成功。

最窄证据结论是：10-shot support-internal baseline 的后期 recovery 跨 backbone training seed 重复性很强；Day14 保护呈稳定的 architecture 差异——VarCNNDirection 重复通过，DF 三 seed 重复未过。故整套跨 architecture 强主张不成立。

## 四次训练与 checkpoint provenance

训练数据、输入、模型、AdamW(lr=.001, weight decay=.0001)、batch 64、30 epochs 与 source-valid macro-F1 选模均与正式 seed 6238 一致。

| Architecture | training seed | best epoch | source-valid accuracy | macro-F1 | checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|
| DF | 6238（复用） | 29 | 72.407% | 71.655% | `1bf85127…c9133ae` |
| DF | 1013 | 28 | 71.204% | 70.670% | `7f6b3a4d…314fbd2` |
| DF | 2024 | 28 | 73.426% | 72.932% | `c9326ede…452f5c` |
| VarCNNDirection | 6238（复用） | 23 | 69.630% | 69.142% | `fc3ade79…1ff83` |
| VarCNNDirection | 1013 | 21 | 69.583% | 69.425% | `a1773eea…d8b4e6` |
| VarCNNDirection | 2024 | 25 | 69.120% | 68.762% | `cbd0f16f…cac866` |

四个新增 checkpoint 的完整路径、SHA-256、30-epoch history、source accuracy/F1、模型文件 hash 与 training-config hash 分别在 `artifacts/*_checkpoint.json`。DF 三 seed source-valid macro-F1 range 2.262 pp，VarCNNDirection 0.663 pp；均未触发预注册的“range>4 pp 或任一 seed 偏均值>2 pp”source-quality 异常标记。

各新增 checkpoint 独立、仅用 v3 supervised-train 拟合 G-source，并只用 official source validation 从 C={.1,1,10} 选择；四者均选择 C=10。对应 source-valid linear macro-F1 为 DF 72.960/74.902%、VarCNNDirection 72.802/71.496%。

## 10-shot 主结果

单元格为三个 support seeds 的 selected baseline `accuracy / macro-F1` 均值。

| Architecture | training seed | Day14 | Day90 | Day270 |
|---|---:|---:|---:|---:|
| DF | 6238 | 72.56 / 72.39 | 63.72 / 63.25 | 58.97 / 58.15 |
| DF | 1013 | 71.32 / 71.16 | 62.77 / 62.29 | 59.19 / 58.39 |
| DF | 2024 | 72.18 / 71.99 | 64.04 / 63.65 | 59.47 / 58.81 |
| VarCNNDirection | 6238 | 70.94 / 70.83 | 62.80 / 62.24 | 60.41 / 59.74 |
| VarCNNDirection | 1013 | 71.38 / 71.27 | 63.20 / 62.55 | 61.18 / 60.54 |
| VarCNNDirection | 2024 | 70.98 / 70.87 | 63.81 / 63.16 | 60.88 / 60.47 |

预注册门槛的精确比较：

| Architecture | seed | Day14 selected−stronger-current | Day90 recovery retention | Day270 recovery retention | seed status |
|---|---:|---:|---:|---:|---|
| DF | 6238 | +1.951 pp | 100.0% | 100.0% | fail: Day14 |
| DF | 1013 | +1.701 pp | 100.0% | 100.0% | fail: Day14 |
| DF | 2024 | +1.313 pp | 100.0% | 100.0% | fail: Day14 |
| VarCNNDirection | 6238 | +4.881 pp | 92.49% | 95.55% | pass |
| VarCNNDirection | 1013 | +4.617 pp | 83.70% | 96.18% | pass |
| VarCNNDirection | 2024 | +4.297 pp | 98.31% | 100.0% | pass |

DF Day90/270 的 18 个配置全部选 G-current，因此 recovery 为 100%。VarCNNDirection 后期在 G-current/current prototype 与 alpha=.50/.75 间选择，仍跨 seed 全部超过 80%。Day14 的 DF 选择随 representation seed 从 G-source 扩展到 shrinkage/interpolation，但没有一个 training seed 达到保护门槛；VarCNNDirection 始终落在 G-source/强 shrinkage 一侧并通过。

10-shot selected 相对 G-source 的 `corrected / harmed / net` 三 support-seed 平均计数：DF seed 6238/1013/2024 在 Day14 为 0/0/0、73.3/75.7/-2.3、291.3/352.3/-61.0；Day90 net 为 +1775.3/+1795.7/+1761.0；Day270 为 +2318.7/+2518.7/+2466.7。VarCNNDirection Day14 net 为 +11.7/-15.3/+7.0；Day90 为 +1510.3/+1460.3/+1618.0；Day270 为 +2438.3/+2596.3/+2611.3。逐 support seed 的 accuracy、macro-F1、corrected/harmed/net 与 102 类指标均在 108 个正式 evaluation JSON 和 `metrics_long.csv`。

## 两类随机性

`randomness_decomposition.json` 对每个 architecture×date×shot×method×metric 保存完整 3×3 cell 表，以及固定 support seed 跨 training seeds、固定 training seed 跨 support seeds 的 mean/SD/range。下表摘要 selected macro-F1；SD/range 是相应三组条件 SD/range 的平均，单位 pp。方差占比来自平衡两因素加性分解，interaction 与 residual 不可分离。

| Architecture | date | shot | training SD/range | support SD/range | training/support/interaction-residual share |
|---|---|---:|---:|---:|---:|
| DF | Day14 | 3 | 0.799 / 1.454 | 0.341 / 0.606 | 67.7 / 5.7 / 26.7% |
| DF | Day14 | 10 | 0.676 / 1.232 | 0.204 / 0.365 | 83.9 / 2.8 / 13.3% |
| DF | Day90 | 3 | 1.305 / 2.561 | 0.928 / 1.783 | 52.7 / 0.6 / 46.8% |
| DF | Day90 | 10 | 0.715 / 1.359 | 0.453 / 0.845 | 67.5 / 27.6 / 4.9% |
| DF | Day270 | 3 | 0.815 / 1.545 | 0.814 / 1.501 | 38.1 / 42.6 / 19.3% |
| DF | Day270 | 10 | 0.424 / 0.801 | 0.411 / 0.775 | 38.2 / 33.3 / 28.5% |
| VarCNNDirection | Day14 | 3 | 0.370 / 0.722 | 0.235 / 0.446 | 69.2 / 22.7 / 8.0% |
| VarCNNDirection | Day14 | 10 | 0.244 / 0.445 | 0.081 / 0.160 | 89.4 / 6.5 / 4.1% |
| VarCNNDirection | Day90 | 3 | 1.085 / 2.111 | 0.809 / 1.590 | 52.5 / 11.4 / 36.1% |
| VarCNNDirection | Day90 | 10 | 0.937 / 1.680 | 0.898 / 1.575 | 13.4 / 43.7 / 42.9% |
| VarCNNDirection | Day270 | 3 | 1.205 / 2.157 | 1.095 / 2.049 | 42.2 / 19.7 / 38.1% |
| VarCNNDirection | Day270 | 10 | 0.615 / 1.197 | 0.864 / 1.613 | 17.8 / 63.5 / 18.7% |

这明确区分了两种随机性。Day14 10-shot 的 selected 变异主要来自 backbone training seed，但规模仍小且 architecture 内过线状态一致；后期 support sampling 在 VarCNNDirection 尤其明显，仍未改变全部 late-date cell 通过的结论。不能用三个 support seeds 代替 backbone-seed 重复。

## 3-shot 附加诊断

3-shot 不参与 replication 主判定，也没有据其结果改方法。跨两个 architecture×三个 training seeds×三个 dates×三个 support seeds 的 54 个配置，selected 与 posthoc oracle 一致 12/54（22.2%），macro-F1 regret 0.835±1.032 pp、range 4.993 pp。10-shot 为 34/54（63.0%），regret 0.220±0.522 pp、range 2.278 pp。因此扩展到 training seeds 后仍重复了既有低标签 selection-noise 结论。

3-shot selected−G-source 的 seed-mean 范围：Day14 在六个 architecture/training-seed 单元均为 -0.488 至 -0.002 pp；Day90 为 +0.277 至 +2.010 pp；Day270 为 +4.398 至 +6.476 pp。它仍明显没有达到常规 10-shot 的后期恢复幅度，且 Day14 主要靠不更新/强保留维持。

## 失败归因与边界

- **不是 source checkpoint/representation quality 显著变化。** 两 architecture 都未触发预注册 source-quality 异常阈值；DF source-valid 甚至包含高于原 seed 的 2024，但 Day14 仍失败。
- **不是少数 date/support-seed 单元波动。** 失败集中在 DF Day14，并在三个 training seeds 的 support-seed 均值上全部重复；相对 stronger-current 的优势随 seed 从 1.951 降到 1.313 pp，没有一次过 2.0 pp。
- **也不是简单的 training-seed 过线敏感。** architecture 内的 pass/fail 模式跨三 seed 完全一致。更准确的描述是 support-selected baseline 的 Day14 保护具有 architecture/representation-family 差异：VarCNNDirection 稳定通过，DF 稳定不足；training randomness 改变效应大小但没有改变结论。后期收益则对 representation training seed 稳健。

本实验不据此设计新 adapter、loss、meta-learning 或其他机制，也不替 Host 作研究路线决定。由于严格 replication 未成立，不把下一步自动收敛为 3-shot 新研究；3-shot 仅作为已验证的低标签瓶颈保留。

## 完整性、异常与产物

- `integrity_check.json`：`passed=true`、0 errors；核验 4 个新 checkpoint、72 个新 evaluation、36 个旧 evaluation hash，并独立重算 19,606,176 条 method prediction 的指标与转移计数。
- `metrics_long.csv`：1,296 行（108 配置×12 方法）；`summary.json` 保存 checkpoint、各 cell support-seed mean/SD/range、18 个主判断与 selection noise；`randomness_decomposition.json` 保存全部 training/support mean/SD/range 与 288 个两因素表。
- 评价期间 GPU 2 被外部进程占满，VarCNNDirection seed 2024 在任何正式 prediction artifact 写出前 OOM；随后以相同 checkpoint、候选、solver、迭代和顺序改用 CPU frozen embedding extraction 完整重跑。另为降低 BLAS 线程争用，中止了两个尚未写出当前 date artifact 的评价进程，随后按同配置恢复。训练未异常，seed 未替换；中断产物不存在。
- 所有新增 code/checkpoint/artifact/log 仅位于本 run；旧实验只读且正式哈希未变化。TemporalDrift query 标签仅在每个 architecture×training-seed×date 的六套预测全部固定后用于评分。
