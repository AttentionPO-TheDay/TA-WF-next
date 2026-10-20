# 来源与修复记录

模型基础结构和方向前端显式复制自 `runs/20261014T070454Z_tent_baseline_v3_83d48bf3` 的 run-local 文件；Proteus/no_pseudo 适应定义参考已审计 `runs/20261009T085657Z_proteus_mechanism_baseline_b5571868` 与 `runs/20261013T091631Z_mvi_dynamic_bn_tta_a79d0c0b`，不导入旧运行时。数据路径由 `configs/datasets.json` 定位；结构审计来自 `runs/20261015T023855Z_external_transfer_audit_e405d268`。

v1 因 `fcntl.flock` 缺少 `fcntl.LOCK_EX` 参数失败；v2 修复锁调用但在 `proteus_default` 的伪标签批次处将 `(x,y,uid)` 错误解包为两项。v3 只修复该实现错误，统一批次接口并加入预检契约检查与单元级断点保护，不使用 v2 的性能反馈调参。
