# 实验结果

## 最终裁决

`STOP_NO_USABLE_INTERFERENCE`

这是 TemporalDrift development evidence。第一关未同时满足预注册的 cross-damage 与 mixed-cancellation 必要条件，因此严格停止；第二关的无标签信号文件未生成。

## 第一关结果

固定 DF/Day90/source-statistics Tent 下，更新本身有效，但没有达到“适合自组而稳定损害跨组”的保守门槛。全部 102 个网站方向的独立 evaluation 结果如下（accuracy 为绝对比例差）：

| 冻结条件 | 观察值 | 预注册要求 | 通过 |
|---|---:|---:|---|
| mean self accuracy delta | +0.012970 (+1.297 pp) | >= +0.005 | 是 |
| mean self true-class probability delta | +0.014057 | >= +0.002 | 是 |
| self accuracy positive rate | 72/102 = 70.59% | >= 60% | 是 |
| mean cross accuracy delta | -0.002500 (-0.250 pp) | <= -0.005 | 否 |
| mean cross true-class probability delta | -0.001847 | <= -0.002 | 否 |
| cross accuracy negative rate | 41/102 = 40.20% | >= 60% | 否 |
| mean mixed accuracy delta | +0.005779 (+0.578 pp) | <= 0.5 × mean self (=+0.006485) | 是（均值项） |
| positive-self 中 mixed cancellation rate | 32/72 = 44.44% | >= 60% | 否 |
| self-positive/cross-negative joint rate | 31/102 = 30.39% | descriptive | — |

Oracle interference contrast `mean(self−cross)` 为 +1.547 pp，高于五个 random repeats 的最大值 +0.0056 pp；同向 joint rate 30.39% 也高于随机最大值 6.86%。这说明网站分组确实产生超出普通随机拆分的异质性，但它不等价于预注册所要求的稳定 cross damage。

## 拼接指标与类别级影响

对全部 22,071 个隔离 evaluation predictions 拼接后计算（不是单网站 F1 平均）：

| 条件 | Accuracy | Macro-F1 | Mean true-class probability |
|---|---:|---:|---:|
| No update | 0.641430 | 0.617803 | 0.598193 |
| Self update | 0.654343 | 0.630348 | 0.612181 |
| Cross update | 0.638938 | 0.615544 | 0.596356 |
| Mixed update | 0.647139 | 0.623510 | 0.604875 |

类别级转移总数：self 为 wrong→correct 326、correct→wrong 41、wrong→different-wrong 331、unchanged-correct 14,116；cross 为 75/130/237/14,027；mixed 为 164/38/198/14,119。No update 的 unchanged-correct 为 14,157，其余转移为 0。

## 简单解释排除与限制

有效筛选比例和相对更新幅度通过冻结审计：三源 median 有效比例最大差 0.0313，relative-delta median 比 1.234；interference contrast 与二者的 Spearman rho 为 0.026/0.273。全部 oracle/random 更新都通过 support entropy fallback。故当前差异不主要表现为“某组根本没更新”或数量级更新幅度差。

限制：只有一个已观察 Day90、一个训练 seed/checkpoint、一种 raw-timestamp DF/Tent 配置；网站配对是覆盖性相邻标签 pairing，不是语义相似性设计；五个 random controls 是同一 checkpoint 下的拆分重复，不是训练重复。Oracle 网站身份不可部署。结果不外推到其他模型、日期、current-batch BN、continual TTA 或外部测试。

## 完整性

新训练 0、backbone 修改 0、新 checkpoint 0、外部测试 0。最终完整运行含 918 个从 source checkpoint 重置的临时 TTA updates；因首次输出后仅补显式 No-update/幅度列而以完全相同冻结配置重跑一次，总执行 1,836 个临时 updates。未据结果改变任何实验选择。
