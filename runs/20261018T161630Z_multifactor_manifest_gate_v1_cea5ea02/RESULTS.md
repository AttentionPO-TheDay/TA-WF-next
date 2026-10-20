# 多因素身份门槛审计结果

CPU 只读完成。逐行清单在 `artifacts/factor_rows.csv`；无训练、适应、预测或性能评分。纯净源域 18,553 行与本轮10个目标文件均无完整原始 trace 精确重叠；这不证明 session 独立或网站语义一致。

| 切片 | 行数 | 102类覆盖 | 源精确重叠 | 可归因/备注 |
|---|---:|---:|---:|---|
| VersionDrift/045/drift.npz | 75034 | 102 | 0 | 唯一版本 65017，未知 10017；唯一归因行均复用train/valid锚点 |
| VersionDrift/046/drift.npz | 80925 | 102 | 0 | 唯一版本 70908，未知 10017；唯一归因行均复用train/valid锚点 |
| VersionDrift/047/drift.npz | 80121 | 102 | 0 | 唯一版本 70104，未知 10017；唯一归因行均复用train/valid锚点 |
| VersionDrift/048/drift.npz | 74423 | 102 | 0 | 唯一版本 64406，未知 10017；唯一归因行均复用train/valid锚点 |
| NetworkDrift/DE.npz | 4217 | 102 | 0 | 仅地域文件名，丢包/延迟数值未证实 |
| NetworkDrift/JP.npz | 3125 | 102 | 0 | 仅地域文件名，丢包/延迟数值未证实 |
| NetworkDrift/SG.npz | 1994 | 102 | 0 | 仅地域文件名，丢包/延迟数值未证实 |
| NetworkDrift/UK.npz | 3618 | 102 | 0 | 仅地域文件名，丢包/延迟数值未证实 |
| NetworkDrift/USA.npz | 3805 | 102 | 0 | 仅地域文件名，丢包/延迟数值未证实 |
| BehaviorDrift/subpage.npz | 20400 | 102 | 0 | URL 17739 个，副本 2661 |

裁决：多因素目标评分**仍未解锁**。Version 可归因行是版本训练/验证锚点重用，未知行不能命名版本；Network 缺定量物理条件与站点身份；Behavior URL 一致性不能替代 homepage↔subpage 网站映射。后续可依据此清单设计已观察数据上的诊断子集，但须另行冻结标签映射和 reference/query 权限，不能称独立盲测。

输入与清单 SHA256、逐文件计数及门槛布尔值见 `artifacts/audit.json`。
