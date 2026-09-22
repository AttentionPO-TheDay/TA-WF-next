# AUDIT_PLAN v1

版本：v1，2026-09-17（Asia/Shanghai）。本计划约束本 Job 的全部审查与裁决。

## 唯一问题与判定

只判断历史 source 是否自带可验证的真实变化条件，使同一网站能跨至少两个条件形成 3-shot support→独立 query episode。真实条件必须来自有来源说明的采集日期、session、capture/run、网络/采集环境、原始文件批次或等价标识；随机 split、行号、包位置、随机增强和单条 trace 内相对包时间均不算条件。

- PASS：条件语义可信、网站覆盖充分、每侧原始来源可隔离且无明显泄漏。
- CONDITIONAL：存在真实条件信号，但覆盖、样本量或 provenance 只支持有限开发。
- FAIL：没有足够证据定义真实变化任务。FAIL 时停止当前版本的表示学习假设，不以伪任务替代。

最低 episode 约束：同一网站至少两个真实条件；support 条件每网站至少 3 条；query 来自独立条件且至少 1 条；原 trace、精确/输入级重复、切片/增强派生族和同 session 派生样本不得跨侧；若有真实时间只允许较早→较晚。覆盖率报告以 102 个网站为分母。

## 权限与范围

- 可读：`train.npz`、`valid.npz` 的 schema/既有 source 统计；当前 split 生成代码与 manifest；正式 provenance、去重/隔离审计；指定三个前序 experiment 的冻结记录。
- 不读/不使用：任何 TemporalDrift future/query `y` 来定义条件、阈值或判断质量。执行中一次全目录 schema 命令误触数组加载并超时、没有返回内容；因此不能证明零物理访问，但没有任何 future 内容进入条件、阈值、统计或裁决。该偏差记录在 `execution_record.md`。
- 不做：训练、微调、解冻、embedding 提取、GPU 作业、数据或 checkpoint 修改、新 loss/网络设计。
- 允许写入：仅本 experiment 的审计文档和 registry/status 记录。

## 权威输入

当前工作区：`AGENTS.md`、`STATUS.md`、`PROTOCOL.md`、`HANDOFF.md`、`configs/datasets.json`、`scripts/run_temporal_screening.py`、`runs/exp_6238dacf9aa142cc/artifacts/{splits_v3.json,split_content_audit_v4.json}`。指定前序记录：`exp_b471517a3e6f41e7`、`exp_04faf4088b604155`、`exp_62c23528b1774202`。来源追溯：旧归档中的 `TemporalDrift_Provenance_Followup_2026-09-12.md` 与正式 Proteus six-dataset protocol audit；旧归档只读引用，不作为运行时代码依赖。

输入哈希与访问记录写入 `summary.json` 和 `execution_record.md`。
