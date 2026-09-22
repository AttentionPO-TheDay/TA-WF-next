# Local Burst Evidence Error-Correction Diagnostic — Correction Plan v1

冻结时间：2026-09-18（Asia/Shanghai）。本计划在本 experiment 的未来日期标签评分前写入。后续只允许记录并修复实现错误，不得依据未来结果更换模型、窗口、特征、模板、尺度、权重、阈值、日期或网站。

## 问题、权限与停止边界

本实验只检验上一阶段冻结的局部 run-length evidence 是否给既有 TemporalDrift 全局模型带来额外纠错价值。TemporalDrift 是已观察的 development benchmark；Day14/30/90/150/270 标签仅用于无标签预测完成后的 metric、错误归因、混淆 pair 和网站/日期汇总。训练、微调、backbone 解冻、Burst CNN、局部神经网络、分类器、selector、learned gate、TTA、current-label adaptation 和自动区域选择均禁止。精确 signed maximal-run RLE 是无损可逆编码，不解释为信息过滤。

只读数据为 `/mnt/data2/ren/datasets/TemporalDrift/{train,valid,day14,day30,day90,day150,day270}.npz`。正式 source manifest 固定为 `runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`（SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`）。所有新产物只写入本 run。

## 冻结全局模型与预测来源

主模型固定为 `runs/exp_9121b664a1854097/checkpoints/df_best.pt` 的 DF seed 6238、epoch 29、原分类头 A。checkpoint SHA-256 为 `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`。它是当前项目正式、可复现、与 v3 source split 和 5000-packet sign 输入一致的普通 source-only baseline；只以 official source validation macro-F1 选模。选择 DF 而非同批 VarCNNDirection 只依据已经正式记录的历史 validation macro-F1（DF 0.71655，高于 0.69142），不依据本 Job 的 future 表现。

既有五日期 prediction JSON 只保存 top-1，不能提供 global rank 或分数融合所需的 102 类分数。因此固定 checkpoint 将以 eval/inference mode 对 official valid 和五个 future 文件各运行一次 forward，保存 float32 logits；不得更新参数。既有 top-1 prediction cache 用于逐行对齐校验：source filename、row index、样本数、truth scoring array 与 argmax 必须全部一致，否则评分停止。全局 score 直接为 logits。

## 冻结局部结构、观察预算与缺失规则

统一输入先取 raw packet index 0..4999，逐元素 `sign`，非零有效前缀后的连续零是 padding。所有候选共同要求前 500 个 raw positions 都是有限非零方向，因而共享完全相同的 500-packet observation budget；不足 500、出现内部零或非有限值时，该样本 local evidence 记缺失，融合严格保持 global logits/prediction。覆盖率单独报告，不删除这些样本的总体 global/fusion accuracy。

在恰好前 500 个方向上构造连续相同 `+1/-1` 的 maximal runs。每个 run 保存符号和精确整数长度；decode 必须逐元素还原 500-direction input，run 长度和为 500、相邻符号交替。前 500 的最后一个 run 按上一阶段 retained-terminal 主定义保留，并标记为可能 right-censored；所有窗口和统计使用同一处理。主窗口固定为零基 raw indices 50..99。设 run `r` 长度为 `l_r`，与半开窗口 `[50,100)` 的重叠 packet 数为 `o_r`，主标量固定为 `sum_r(o_r*l_r)/sum_r(o_r)`。不得搜索位置或改变公式。

## Day0-only robust class scoring

模板只使用 v3 `supervised_train` 的 16,309 条历史有标签样本；`reference` 和 `source_holdout` 不并入模板，official `valid.npz` 仅用于下述融合权重选择。对每一候选标量、每类 `c`，模板 `m_c` 为 eligible train 值的 median，原尺度为 `1.4826 * median(|x-m_c|)`。尺度下限固定为 102 个 class raw scale 中正有限值的 median；最终 `sigma_c=max(raw_scale_c, pooled_positive_median, 1e-6)`。若某类无 eligible train 值则实验停止。样本 `x` 的 local class score 为 `L_c=-|x-m_c|/sigma_c`。

报告 rank 时按 score 降序、class id 升序确定完全可复现的 ordinal rank；top-k 使用同一顺序。local margin 固定为 `L_true-L_global_pred`。缺失 local evidence 的 rank/margin 为空，不用任意 class-id tie 伪造证据。

## Day0-only 静态融合

对每个 eligible 样本，global logits 和 local score 分别在该样本的 102 类内减均值并除以 population standard deviation（下限 `1e-12`），得到 `Gz` 与 `Lz`。静态融合为 `F=(1-alpha)Gz+alpha*Lz`。候选 `alpha` 在本计划固定为 `[0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]`；只在 official Day0 `valid.npz` 上最大化 top-1 accuracy，完全并列取较小 alpha。若 local score 跨类零方差或样本不 eligible，则该样本 `F=Gz`。不使用 conflict margin、future batch 统计、温度搜索或未来标签。主规则是这一完全静态版本。

## 冻结特异性对照

所有对照使用相同 500-packet eligibility、RLE/terminal/padding 规则、robust class scoring、alpha grid 和 Day0 valid 选择准则；每个对照可由相同 Day0-only 规则公平选择自己的 alpha，不共享未来信息。

1. `run_mw_0_49`：与主窗口等宽的相邻早期窗口 `[0,50)`，同一 covering-run mass-weighted length。
2. `run_mw_100_149`：等宽窗口 `[100,150)`，同一统计。
3. `direction_out_fraction_50_99`：主窗口 outgoing (`+1`) packet fraction；incoming fraction 是其补数并一并在定义中说明，不重复当作第二个独立候选。
4. `direction_transition_density_50_99`：主窗口相邻方向转换数除以 49。
5. `prefix150_out_fraction`：固定 `[0,150)` 的 outgoing packet fraction。
6. `prefix150_transition_density`：固定 `[0,150)` 的相邻方向转换数除以 149。

这不是窗口搜索；无候选会在看 future 结果后替换或组合。主统计名固定为 `run_mw_50_99`。

## 诊断与输出

`local_rank_diagnostic.csv` 对每日期和 all/global-correct/global-wrong 子集报告 eligible/total 数、true-class local rank mean/median、top-1/5/10 hit、local margin mean/median，并含总体汇总。`error_complementarity.csv` 仅在 global-wrong 中按 overall/date/site/实际 true→pred pair 报 global/local rank、improved/unchanged/worsened、推入 top-k 和 local 更支持 true/pred；pair 只归因，不进入规则。`fusion_repair_table.csv` 对主统计及全部对照按日期与总体报告 wrong→correct、correct→wrong、wrong→different-wrong、unchanged-correct、其他 unchanged、Net Repair、比例、global/fused accuracy 与 delta。`specificity_controls.csv` 汇总相同量用于主统计与对照比较。`per_site_date_breakdown.csv` 报主统计的逐网站×日期覆盖、rank、互补性与修复。

另生成 `global_prediction_audit.md`、`local_scoring_definition.md`、`RESULTS.md`、`summary.json` 和 `execution_record.md`。所有 CSV 保留明确 `development_evidence=1` 标记。

## 保守裁决规则

技术裁决严格限定为三类，不替 Host 作研究路线决定：

- `SUPPORTS_CONFLICT_RELIABILITY_FOLLOWUP`：主证据在至少 4/5 future 日期获得 `Net Repair>0`，总体 `Net Repair>0`；global-wrong eligible 样本中 local rank improved 比例高于 worsened；正净修复覆盖至少 3 个日期且至少 20 个网站；并且主统计总体 Net Repair 严格大于每个相邻窗口和每个普通 direction/prefix control，且至少比最强 control 多 `max(10, 20%)` 个净修复。该阈值只作保守 gate，不声称创新成立。
- `EARLY_TRAFFIC_ONLY`：主统计总体 `Net Repair>0` 且至少 3/5 日期为正，但上述特异性优势不成立，或最强对照达到主统计的 80% 以上；只支持 early-traffic complementarity。
- `STOP_LOCAL_CORRECTION_ROUTE`：其余情况，包括总体 `Net Repair<=0`、少于 3 个日期为正、正净修复少于 20 个网站，或 global-wrong local rank improved 不高于 worsened。

完成固定产物与校验后停止，不创建任何后续训练、selector 或 TTA Job。
