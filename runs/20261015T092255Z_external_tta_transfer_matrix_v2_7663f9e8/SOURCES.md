# 来源与修复记录

模型基础结构显式复制自 `runs/20261014T070454Z_tent_baseline_v3_83d48bf3` 的 run-local 文件；Proteus/no_pseudo 适应定义参考已审计 `runs/20261009T085657Z_proteus_mechanism_baseline_b5571868` 与 `runs/20261013T091631Z_mvi_dynamic_bn_tta_a79d0c0b`，不导入旧运行时。数据路径由 `configs/datasets.json` 定位；结构审计来自 `runs/20261015T023855Z_external_transfer_audit_e405d268`。

首个矩阵 run `20261015T030319Z_external_tta_transfer_matrix_12f3d902` 在启动阶段失败：`fcntl.flock` 缺少 `fcntl.LOCK_EX` 参数；没有产生性能结果。v2 仅修复 `runner.py` 中的锁调用，并将审计实际确认的 AWF 共同标签数 189 写入配置；没有利用性能反馈。
