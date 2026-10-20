# 外部 TTA 迁移性矩阵 v3

## 问题与修复设计

本 run 复测冻结的 WTT-Time/AWF 方向前端迁移矩阵。v2 在首个 `proteus_default` 单元因伪标签 DataLoader 三字段批次被错误按两字段解包而失败。v3 保留 v2 的数据、模型、超参数、条件、输入前端、标签权限和预算；仅做实现错误修复与执行保护：

- 统一 `next_batch(iterator, loader)` 接口，显式验证每个批次为 `(x, label, uid)` 三项，循环耗尽时重建迭代器；
- 修复 Proteus 伪标签批次解包；
- 启动前以预检验证数据、模型前向和批次契约；
- 每个单元完成后立即写入 `scores.json`、预测和轨迹，并支持按 `(dataset, seed, slice, condition)` 跳过已完成单元，避免中断后重复计算；
- 不改变算法、超参数、数据划分、选模规则或评价权限。

## 冻结比较

比较 `frozen`、`tent_bn`、`tent_entropy_gate`、`proteus_default`、`no_pseudo_fixed` 五个条件。两套数据集分别训练自己的 wide+BN 模型，输入为方向序列前端；query 标签只用于终端评分。

## 数据与权限

WTT-Time 使用 15 个历史日期训练、5 个开发日期和 7 个 future 日期评分；future 结果属于已暴露诊断，不能作为未见确认。AWF 使用 source 与五个 query 文件共同的 189 类，首批 1024 条作无标签 reference，其余作 query，属于诊断评价。不得使用 query 标签适应或选模。

## 预算与停止

3 个训练 seed、255 个数据集×slice×condition 单元；source 45 epoch，reference 1024 条，800 次候选更新；GPU 1，单进程，最大 wall time 172800 秒。失败时保留 failure.json、日志和已完成单元；实现修复复测另立 run。
