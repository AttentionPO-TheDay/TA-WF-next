# 实验结果

## 完成状态

有界表示诊断已按预注册 source→freeze→future 顺序完成。新增 backbone 训练为 0；未使用外部数据集、DNNF/TFAN、attention、新损失、额外层或大规模超参搜索。

## 冻结 source 选择

- df: `intermediate_ordered4__linear`，C=10.0；validation accuracy/F1 53.10/51.90，source-holdout 53.53/52.53。
- varcnn_direction: `intermediate_ordered4__linear`，C=10.0；validation accuracy/F1 42.87/42.07，source-holdout 45.98/45.11。

完整 source 矩阵、门槛判断、逐网站指标与监督预算限定见 `SOURCE_RESULTS.md`、`artifacts/source_matrix.csv` 和两个 `*_source_diagnostic.json`。

## Future 绝对结果、相对 source 下降与 C 差值

单元格均为 accuracy / macro-F1 百分数或百分点。下降为 future-source，因此负数表示下降。

| Backbone | 日期 | 冻结诊断 | 相对 source | 旧 C | 诊断-C |
|---|---|---:|---:|---:|---:|
| df | day14 | 52.86 / 52.00 | -0.66 / -0.53 | 53.31 / 52.34 | -0.45 / -0.35 |
| df | day30 | 49.29 / 48.33 | -4.24 / -4.20 | 49.29 / 48.09 | 0.00 / 0.25 |
| df | day90 | 41.45 / 39.63 | -12.08 / -12.90 | 41.58 / 39.47 | -0.13 / 0.16 |
| df | day150 | 37.05 / 35.00 | -16.48 / -17.53 | 37.84 / 35.30 | -0.79 / -0.30 |
| df | day270 | 33.27 / 31.09 | -20.26 / -21.44 | 34.70 / 32.01 | -1.43 / -0.92 |
| varcnn_direction | day14 | 44.26 / 43.58 | -1.72 / -1.53 | 41.57 / 40.45 | 2.69 / 3.13 |
| varcnn_direction | day30 | 40.16 / 39.45 | -5.82 / -5.66 | 38.64 / 37.65 | 1.52 / 1.79 |
| varcnn_direction | day90 | 33.17 / 31.83 | -12.81 / -13.28 | 32.95 / 31.69 | 0.22 / 0.14 |
| varcnn_direction | day150 | 30.41 / 27.69 | -15.57 / -17.41 | 29.45 / 27.54 | 0.95 / 0.16 |
| varcnn_direction | day270 | 26.82 / 24.78 | -19.16 / -20.32 | 28.54 / 26.42 | -1.73 / -1.64 |

## 按预注册门槛的证据解释（不替 Host 决策）

- df: 浅层 global mean 未达可用门槛；ordered linear 显著优于对应 cosine=True；唯一中间层显著优于浅层=True；future 相对 C 至少 4/5 日期 macro-F1 为正=False。
- varcnn_direction: 浅层 global mean 未达可用门槛；ordered linear 显著优于对应 cosine=True；唯一中间层显著优于浅层=True；future 相对 C 至少 4/5 日期 macro-F1 为正=True。
- 浅层 ordered cosine 相对旧无序 D 的 holdout macro-F1 差为 DF +5.23 pp、VarCNNDirection +3.44 pp；按 5 pp 门槛仅 DF 明确支持顺序信息。浅层 ordered linear 相对 global linear 则分别 +24.04/+25.09 pp，说明保留粗位置对监督读出很重要，但不能单独归因于“有序优于无序”，因为读出预算不同。
- 浅层 ordered linear 相对同表示 cosine 为 +24.31/+25.13 pp；中间层 ordered linear 相对浅层为 +18.03/+10.43 pp。两 backbone 的模式一致：原浅层表示与 2-reference 余弦读取都构成瓶颈，唯一中间层能恢复一般 source 识别能力。
- 两 backbone 都有 validation F1≥40% 的 source 配置：True；两者都达到重复 future 正向收益：False。只有二者同时为 True 才满足建议 DNNF/TFAN 同权限比较的证据条件。
- ordered 对无序的 5 pp 证据按 backbone 分开报告；即使为正也只说明有限顺序结构值得检验，不证明局部抗漂移机制。
- 本实验属于“source 恢复、但两个 backbone 未共同获得相对 C 的重复未来优势”：DF 仅 2/5 日期 macro-F1 略高于 C，VarCNNDirection 为 4/5，但 Day270 反转。按预注册框架，这些证据应归为一般识别问题的修复，仍不支持局部抗漂移假设，也未满足进入 DNNF/TFAN 同协议比较的证据条件。是否据此结束当前局部候选由 Host 决定。

## Provenance、成本与限制

- Checkpoint：DF epoch 29 / SHA-256 `1bf851...c9133ae`；VarCNNDirection epoch 23 / `fc3ade...1ff83`。完整路径、哈希、训练记录与模型来源见 `representation_diagnostic_plan.md` 和 source JSON。
- Split v3 SHA-256 `0f322e...3f142`，16,309 train / 204 reference / 2,040 holdout；official valid 2,160。reference/holdout 未拟合或选模。
- VarCNNDirection 中间层对短流量有明显覆盖限制；所有短样本以预注册零表示保留，没有过滤。完整覆盖和运行成本见 `artifacts/layer_source_audit.json` 及各 JSON。
- Future 日期是已观察的 TemporalDrift 开发证据，不是独立确认。线性读出监督预算远大于 2-reference 方法，任何提升不归因于原局部机制。
- 每 backbone 的 future JSON 和 predictions JSON 保存完整逐网站结果与固定预测；旧 C 来自 donor artifact，并标记为非本次重跑。
