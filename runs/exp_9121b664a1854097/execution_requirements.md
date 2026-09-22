# Execution requirements

## 协议来源与核验

- 项目约束：`/home/rbf/TA-WF-next/AGENTS.md`、`STATUS.md`、`PROTOCOL.md`、`HANDOFF.md`，均已在训练、评价和代码修改前完整读取。
- 冻结执行规范：`/home/rbf/TA-WF-next/runs/exp_6238dacf9aa142cc/PLAN.md` 与 `RUN_COMMANDS.md`，均已完整读取。旧 experiment 仅为协议/artifact 来源，不继承其运行状态。
- 正式 split：只读复用 `/home/rbf/TA-WF-next/runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`，schema v3，SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`；JSON 可读，102 类均覆盖。不得生成或退回 v1/v2。
- split 内容复核：只读 `/home/rbf/TA-WF-next/runs/exp_6238dacf9aa142cc/artifacts/split_content_audit_v4.json`，`pass=true`，源角色间及与 official valid admitted-input 交集均为 0。
- 局部规范：只读 `/home/rbf/TA-WF-next/runs/exp_6238dacf9aa142cc/artifacts/local_layer_spec_v4.json`，revision 4，SHA-256 `a38b31b4b4ffe5256680821f0d9ef7225741098daae84c708d180f92eb8e6720`。

## 数据、split 与输入

- 数据根 `/mnt/data2/ren/datasets/TemporalDrift`；`train.npz`、`valid.npz`、`day14.npz`、`day30.npz`、`day90.npz`、`day150.npz`、`day270.npz`，键 `X,y`，102 类。
- seed 6238。v3 在每类 canonical source 行的 PCG64 排列中取前 2 条 reference、随后 20 条 source-holdout、其余 supervised-train；实际总数应为 204 / 2,040 / 16,309。
- official `valid.npz` 整体只用于 source-only checkpoint 选择；按 macro-F1 最大值选 epoch，完全并列取最早 epoch。其 9 个内部重复副本作为固定限制，不改变。
- 输入固定为有符号时间戳前 5000 位取 sign，正为 +1、负为 -1、0 保持 0；数值不解释为包大小。DF 输入 `[B,1,5000]`，Var-CNN 使用方向单分支 `VarCNNDirection`，不得称为完整双分支 Var-CNN。

## 训练预算

- DF 与 `VarCNNDirection` 各从头初始化、各一个普通 source-only 监督 seed；B/C/D/E 共享该 backbone 的同一个 best checkpoint，不特殊训练。
- 两者均 seed 6238，AdamW，learning rate 0.001，weight decay 0.0001，batch 64，最多 30 epoch。每 backbone 最多一次新增训练，总新增训练最多 2 次。
- 不复用旧 checkpoint。训练前只执行一次 GPU 利用率、显存和进程冲突检查；不终止或抢占进程。资源不足则保留证据并停止，不能缩短 epoch、减少数据、改 batch、切 CPU 或改变输入协议。

## A/B/C/D/E 与 reference 权限

- A：best checkpoint 原分类头。
- B：L2 归一化全局 embedding；每类同一 2 条历史 reference 均值后再归一化；最大余弦类别。
- C：同一全局 embedding 对全部 204 条 reference 作余弦 1-NN，继承 reference 标签。
- D：固定浅层卷积图；每个位置的完整理论 RF 必须位于 0..4999 且覆盖输入全部非零、有限。合格位置按顺序等分为 4 组，组内均值并 L2 归一化。每个 query 区域对某 reference 的 4 区域取最大余弦，再对 4 个 query 区域求均值，选择最高分 reference。
- E：与 D 同一层、同一合格位置，直接平均原始位置特征并归一化，对同一 reference 作余弦 1-NN；不平均已归一化区域描述子。
- DF 层 `feature_extraction.0`，输出 `[B,32,1249]`，RF=22、stride=4、inclusive RF `[4i-8,4i+13]`，完整长度候选 `i=2..1246`。
- VarCNNDirection 层 `dir_encoder.convs.0`，输出 `[B,64,1250]`，RF=35、stride=4、inclusive RF `[4i-17,4i+17]`，完整长度候选 `i=5..1245`。
- D/E 至少 4 个有效位置；不足则该 query 回退 C。无有效描述子的 reference 同时从 D/E 排除；若任一类失去全部有效 reference，本批所有 query 的 D/E 回退 C。必须保存有效位置数、实际区域推理 mask/标识、回退数，并报告全量与区域推理子集指标。
- reference 只来自冻结 source reference（每类 2），不以未来 query 更新，不用未来标签、伪标签或匹配结果更新。

## 评价、诊断与成本

- 完整评价日期：source-holdout、Day14、Day30、Day90、Day150、Day270。
- Day14/30/90/150/270 标签只在全部无标签预测和匹配选择固定后用于最终开发评分与事后诊断；不得训练、选模、reference 更新、片段选择、阈值/超参数选择或推理决策。
- 每个 backbone × A/B/C/D/E × date 报 accuracy、macro precision/recall/F1、逐网站 accuracy；逐日期报 D-C，以及 future(D-C)-source(D-C)；另报 accuracy/macro-F1 的 D-E 及其相对 source 差。
- D 事后诊断：正确同网站 reference 匹配、错误跨网站匹配、reference/local-region 被不同真网站命中的广度和频率、共同高频非区分性模式；跨 backbone 核对 C 未解决且重复出现的错误模式。诊断不得返调本版本。
- 成本：reference 原始/global/local/same-layer 存储字节、reference 与 query 特征抽取 wall time、A/B/C/D/E 分类合计 wall time、比较次数/等价点积数。

## 输出、冲突与停止条件

- 所有新 checkpoint、日志、预测、指标、诊断和最终报告写入 `/home/rbf/TA-WF-next/runs/exp_9121b664a1854097`；不覆盖旧 experiment 的任何文件。
- 实现冲突：现有 `scripts/run_temporal_screening.py` 将旧 run 目录硬编码为读写目标，而旧 `RUN_COMMANDS.md` 的命令也 tee 到旧日志。为满足新实验隔离，执行前仅增加显式新 run 目录参数；split 和 v4 spec 仍从旧目录只读，所有可变产物转到新目录。不得借此改变研究定义。
- 冻结 PLAN 比用户概括多一个 E 同层对照，且主问题同时包含 D-E；本次按 PLAN 执行 A/B/C/D/E。
- 完成两个 seed 的冻结评价后停止；不自动扩展多 seed、WTT-Time、AWF、DNNF、TFAN、few-shot、额外网格或新机制。若资源、协议或实现阻塞，保存证据并停止，不放宽标签权限或研究问题。
