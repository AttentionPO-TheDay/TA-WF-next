# Source-only representation diagnostic

本报告在任何本实验 future 特征抽取前生成。配置选择只使用 official source validation；source-holdout 只用于下表最终报告。线性分类器使用 supervised-train 的约 160 标签/类，不是与 2-reference 余弦读出的公平部署基线。

## df

| 层/读取 | validation F1 | holdout accuracy / F1 |
|---|---:|---:|
| `intermediate_global__cosine_1nn` | 6.92 | 7.01 / 6.69 |
| `intermediate_global__linear` | 26.95 | 26.76 / 24.75 |
| `intermediate_ordered4__cosine_1nn` | 13.83 | 15.29 / 14.08 |
| `intermediate_ordered4__linear` | 51.90 | 53.53 / 52.53 |
| `intermediate_unordered4__cosine_region_permutation_invariant` | 8.84 | 8.38 / 7.78 |
| `shallow_global__cosine_1nn` | 5.34 | 3.97 / 3.92 |
| `shallow_global__linear` | 10.07 | 13.09 / 10.46 |
| `shallow_ordered4__cosine_1nn` | 11.18 | 10.74 / 10.19 |
| `shallow_ordered4__linear` | 34.36 | 36.37 / 34.50 |
| `shallow_unordered4__cosine_region_permutation_invariant` | 5.68 | 5.54 / 4.96 |

旧冻结对照（非本实验重跑）：A 71.96/71.35, C 52.16/50.88, D 5.54/4.96, E 3.97/3.92（accuracy/F1）。

- 浅层 global mean 可用（holdout F1≥40%）：False。
- 浅层 ordered cosine 相对 unordered 的 F1 差：5.23 pp；达到 5 pp 门槛：True。
- 浅层 ordered linear 相对 global linear 的 F1 差：24.04 pp。
- 浅层 ordered linear 相对 ordered cosine 的 F1 差：24.31 pp；达到监督差异更严格的 10 pp 门槛：True。
- 中间层 ordered linear 相对浅层同读出的 F1 差：18.03 pp；达到 5 pp 门槛：True。

## varcnn_direction

| 层/读取 | validation F1 | holdout accuracy / F1 |
|---|---:|---:|
| `intermediate_global__cosine_1nn` | 7.18 | 6.96 / 6.44 |
| `intermediate_global__linear` | 26.68 | 29.66 / 27.74 |
| `intermediate_ordered4__cosine_1nn` | 7.55 | 8.73 / 6.97 |
| `intermediate_ordered4__linear` | 42.07 | 45.98 / 45.11 |
| `intermediate_unordered4__cosine_region_permutation_invariant` | 5.42 | 6.96 / 5.59 |
| `shallow_global__cosine_1nn` | 5.71 | 5.20 / 5.12 |
| `shallow_global__linear` | 9.08 | 12.84 / 9.59 |
| `shallow_ordered4__cosine_1nn` | 10.60 | 10.34 / 9.55 |
| `shallow_ordered4__linear` | 32.37 | 36.52 / 34.68 |
| `shallow_unordered4__cosine_region_permutation_invariant` | 4.94 | 7.16 / 6.11 |

旧冻结对照（非本实验重跑）：A 68.73/68.48, C 43.04/42.00, D 7.16/6.11, E 5.20/5.12（accuracy/F1）。

- 浅层 global mean 可用（holdout F1≥40%）：False。
- 浅层 ordered cosine 相对 unordered 的 F1 差：3.44 pp；达到 5 pp 门槛：False。
- 浅层 ordered linear 相对 global linear 的 F1 差：25.09 pp。
- 浅层 ordered linear 相对 ordered cosine 的 F1 差：25.13 pp；达到监督差异更严格的 10 pp 门槛：True。
- 中间层 ordered linear 相对浅层同读出的 F1 差：10.43 pp；达到 5 pp 门槛：True。

