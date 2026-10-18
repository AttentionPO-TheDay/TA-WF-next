# 实验依据的 Git 归档规则

本仓库保存代码、实验计划和结论，并选择性保存支撑结论的小型结果与审计文件。`evidence_manifest.json` 列出本次归档的原路径、大小、SHA-256 和类别；原文件字节不改写。项目的 `.gitattributes` 禁用文本换行转换，避免改变冻结文件的哈希。

## 本次选择

`scripts/archive_evidence.py` 的 `compact-evidence-v1` 策略仅检查有 `PLAN.md` 的本项目 run，选择：

- 具名的评分、汇总、对照、审计、验证、失败报告及冻结、来源、输入/输出封存记录。
- `metrics.csv`、`metrics_long.csv`、`aggregate_metrics.csv` 等汇总指标表。
- `invalid`、`superseded`、`repair_history`、`pre_revision`、`pre_write_repair` 路径内的本项目历史源码、计划和修订说明；保留原有无效或被替代标记。
- 部分历史结构审计的 Markdown 摘要。

通常单文件不超过 512 KiB；`scores.json` 上限为 4 MiB，以保留包含按 epoch 损失历史的评分文件。CSV 最多 2000 条数据记录且必须包含指标列。JSON 必须能解析；含具名原始输入、标签、预测数组或超过 2000 项数组的候选文件暂不纳入。脚本还检查常见凭据特征；这些自动筛选不等于完整的数据审查。

筛选与结果是否正向无关，失败和阴性记录同样保留。聚合报告内已有的样本行号、类别统计和按 epoch 历史可能保留，但不单独归档逐样本预测或逐步更新轨迹。

## 继续保留在本地的内容

模型权重、NPZ/NPY 预测和数组、原始/共享数据、预处理缓存、运行日志、事件流、进度和锁文件、虚拟环境均继续忽略。第三方源码快照未纳入本次归档。命中大小或内容限制的白名单候选列在 manifest 的 `excluded_candidates` 中；该列表不枚举所有被忽略文件。

Git 归档不是完整实验备份。关键 checkpoint、预测和大型清单仍需另行备份；本次不会上传、移动或删除它们。

## 复核和后续更新

```text
python scripts/archive_evidence.py
python scripts/archive_evidence.py --write-manifest
python scripts/archive_evidence.py --verify
```

默认命令仅显示候选计数；`--write-manifest` 生成当前文件清单；`--verify` 核对清单中每个文件的原始字节哈希和筛选条件。工具不执行训练、适应、指标重算、实验脚本或 Git 暂存操作，也不加载二进制模型/数据。

本次采用显式路径强制暂存选中文件，保留 `.gitignore` 的目录级排除规则。已选文件进入 Git 后，其修改会正常被跟踪；新产物仍需再次筛选。更新归档前，应查看候选变化，并复核新的文件内容与凭据检查结果。

归档只保存已有记录，不重新验证其科学结论，不将运行中或失败 run 认定为完成，也不改变 `EXPERIMENTS.csv`、`STATUS.md` 和各 run 的结果裁决。历史修正版本继续作为历史证据，不能用来替代已冻结的当前实现。
