# Token 对齐：源域准备结果

三 seed 的纯净源 Token 统计量、102 类原型和逐行 pooled embedding 已完成；每 seed 仅使用锁定的 18,553 行，底座参数与 BN 缓冲区均未改变。

目标域 A/B/C 消融仍待切片身份、标签映射及 reference/query 清单审计；本结果不包含 ΔF1 或对齐有效性的结论。目标评分未启动。


后续承接（2026-10-19）：三seed源Token参考已完成并冻结复用；原目标A/B/C阶段由诊断v1七条件矩阵承接，不重复源参考构建。 执行见 `runs/20261018T164924Z_multifactor_diagnostic_v1_4fc8f13d`。旧计划、配置和结果保留，不将新评分记到本run。
