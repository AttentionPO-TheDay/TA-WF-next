# 旧研究交接：只保留影响新工作的事实

归档位置：`/home/rbf/TA-WF`。原始论文、代码、配置、日志、checkpoint、审计和结果均留在原处，不删除、不批量改名。本摘要不是旧项目的完整结果目录。

## 已经得出的结论

| 工作 | 已知结论 | 原证据 |
|---|---|---|
| WTT 数据处理 | 100 类、27 日期、15/5/7 划分，数据底座完成 | [处理报告](/home/rbf/TA-WF/docs/WTT_Time_数据处理最终结果总结.md) |
| R2 冻结维护 | 本站少标签插值有效，donor-only 平均有损 | [结果](/home/rbf/TA-WF/outputs/wtt_time_frozen_maintenance/exp_0a0ff805c57040b8/RESULTS.md) |
| R3 匹配训练 | 共享修正未通过预定门槛，候选 FAIL/STOP | [结果](/home/rbf/TA-WF/outputs/wtt_time_r3/exp_14b40ff76a0c42e7/RESULTS.md) |
| R3 C/A 归因 | 零标签优势 +1.5374 pp，大于 1/3-shot 优势；不能证明额外标签效率 | [诊断](/home/rbf/TA-WF/outputs/wtt_time_r3/exp_14b40ff76a0c42e7/diagnostic_ca_v1/RESULTS.md) |
| R3-E | 去掉 donor 位移后主要收益保留；原报告正式判定 inconclusive/weak，不是统计等价证明 | [消融](/home/rbf/TA-WF/outputs/wtt_time_r3e/exp_310a0be8aca94001/RESULTS.md) |
| 文献审查 | 少样本表示、训练后适应、局部匹配均有强相关工作；没有已验证的新创新点 | [文献矩阵](/home/rbf/TA-WF/outputs/literature_review_20260914/PAPER_MATRIX.md) |
| Proteus 审计 | 六类设置不是六个日期；公开实现与论文的分项复现存在缺口 | [审计](/home/rbf/TA-WF/outputs/proteus_six_dataset_protocol_audit_exp_d839f7ce217f488a/proteus_six_dataset_protocol_audit.md) |

更早的 Tent/E2-core 和原型权重、历史记忆探索在旧项目中已有收口或负结果；若要复用，应查原实验，不能直接当作未探索的新机制。

## 数据事实与使用历史

Proteus TemporalDrift：Day0/14/30/90/150/270，原始 X 是有符号相对时间戳，不是包大小。旧 tam_day90 存在转换不一致，不能默认作为可信输入。已多轮用于开发。公开代码记录目标真标签下的最佳 F1，但论文是否采用该值尚未闭合，不能直接认定造假，也不能直接把 best 值作为公平无标签基线。

AWF：方向序列，独立于 Proteus，包含 3d/10d/2w/4w/6w 等文件。跨日期类别集合和样本权限使用前按已有审计确认。

WTT-Time：有符号包大小与顺序，无逐包时间戳。固定底座路径 `/mnt/data2/ren/datasets/processed/wtt_time/exp_26dd67e087824c2f`。原项目 data 路径有兼容链接，冻结文件未改写。

Network/Behavior 可能支持各自条件实验；Version 聚合目标缺逐行版本字段；暂不把它们混成统一六域训练数据。局部 trace 匹配能否重建部分版本身份未完成验证，不能把尚未验证的可恢复性说成绝对不可能。

用户选择的是干净的新工作区，不是删除证据或将旧数据重新宣称为未见数据。新的工作区没有自动继承旧 checkpoint、调参配置或 donor 假设。
