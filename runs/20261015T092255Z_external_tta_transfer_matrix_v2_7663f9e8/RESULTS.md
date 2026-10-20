# 外部 TTA 迁移性矩阵 v2 结果

## 状态

执行失败，未完成预定矩阵。进程 `2318084` 已退出；失败信息已保存于 `artifacts/failure.json` 和 `lgs/runner.out`。

## 已完成单元

共完成 3/255 个单元，均为 WTT-Time、seed `21729`、开发日期 `2021-12-12`：

| 条件 | Accuracy | Macro-F1 | 负适应率 | 更新数 | 保留率 |
|---|---:|---:|---:|---:|---:|
| frozen | 78.333% | 75.328% | 0.000% | 0 | 0.000 |
| tent_bn | 78.838% | 75.645% | 1.934% | 800 | 1.000 |
| tent_entropy_gate | 78.081% | 74.545% | 1.805% | 799 | 0.052 |

上述单元的预测、轨迹和 checkpoint 已保留；这些结果只是本次失败 run 的部分产物，不构成完整矩阵结论。

## 失败原因

进入 `proteus_default` 后，`runner.py:118` 执行 `pb, pi = next(pi)`。伪标签批次实际返回三项 `(x, y, uid)`，因此触发：

```text
ValueError: too many values to unpack (expected 2)
```

这是实现错误，不是数据或评价结果。修复和复测需建立新版本 run，并记录修复影响；本 run 的冻结配置和已产物保持不变。

## 权限与范围核对

预检通过；本次没有进入 AWF 或 WTT-Time 其他日期，没有使用 query 标签进行适应或选模，也没有访问 WTT-Time future 日期。完整协议、输入、权限和预算见 `PLAN.md` 与 `config.json`。
