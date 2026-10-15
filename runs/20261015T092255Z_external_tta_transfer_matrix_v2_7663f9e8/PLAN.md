# 外部 TTA 迁移性矩阵 v2

## 修复说明

本 run 复制首个冻结矩阵 `20261015T030319Z_external_tta_transfer_matrix_12f3d902` 的协议、代码和配置。首个 run 在启动阶段因 `status_update` 使用 `fcntl.flock(lock)` 而不是 `fcntl.flock(lock, fcntl.LOCK_EX)` 失败，未训练、未适应、未产生性能结果。v2 仅修正该锁调用；不因任何数据反馈改变算法、超参数、输入前端、数据划分或评价权限。

## 问题与冻结条件

在不重新调参的情况下，比较 WTT-Time 与 AWF 上 `frozen`、`tent_bn`、`tent_entropy_gate`、`proteus_default`、`no_pseudo_fixed` 五个条件。两套数据集分别训练自己的 wide_bn 模型，使用方向序列前端；query 标签只用于最终评分。

## 数据与权限

WTT-Time 使用 15 个历史日期训练、5 个开发日期和 7 个 future 日期评分；future 结果为已暴露诊断，不能作为未见确认。AWF 使用 source 与五个 query 文件共同的 189 类，首批 1024 条作无标签 reference，其余作 query，属于诊断评价。不得用 query 标签适应或选模。

## 预算与停止

3 个训练 seed、255 个数据集×slice×condition 单元；source 45 epoch，reference 1024 条、800 次候选更新；GPU1，单进程。执行成功后保留全部日志、checkpoint、预测和审计产物。
