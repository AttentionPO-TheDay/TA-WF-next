# Local Burst Evidence Error-Correction Diagnostic — Correction Plan v2

冻结时间：2026-09-18（Asia/Shanghai）。本修订发生在任何本 Job future 文件、future logits 或 future 标签评分被读取之前。v1 的 source-only `prepare` 按预注册规则停止，因为 v3 Day0 `supervised_train` 中 class 4、8、23、43 在共同前 500 packet 预算下各有零条 eligible trace，未生成参数或 future 结果。Host 明确授权以下 source-only neutral-evidence 修订。除本文件明确替换的条款外，`CORRECTION_PLAN_v1.md` 全部约束继续有效。

## 修订原因与不可改变项

Day0 source-only 覆盖审计显示：在 v3 `reference`、`source_holdout`、`supervised_train` 三角色并集上，class 4、8、23、43 仍各有零条满足前 500 raw positions 均有限非零的 trace；official valid 也不能补齐这些类。因此缺失由冻结的 **500-packet common budget** 导致，不是 50–99 window 自身导致。不得放宽 500-packet budget、改用短 trace、移动窗口、改变 covering-run statistic、借用其他类模板或读取 future 标签解决缺失。

主窗口仍为 zero-based raw indices 50..99，主统计仍是在恰好前 500 directions 的 maximal runs 上计算 `sum(overlap_packets * observed_run_length)/50`。terminal/right-censoring、padding、有效长度、RLE round-trip 与全部 controls 保持 v1 定义。全任务仍为 102-class；四类不得删除或从总体 metric 排除。

## Source-defined class support 与 neutral local contribution

主统计及全部 controls 的 supported class set 固定为具有至少一条 v3 `supervised_train` eligible trace 的类；source-only 审计给出 `S={0..101}\{4,8,23,43}`，unsupported set 固定为 `U={4,8,23,43}`。不得为 `U` fabrication、borrow、interpolate 或 estimate class template。`S` 中每类的 median、MAD 与 pooled positive scale floor 仍按 v1 拟合。

对 eligible query，先对每个 supported class 计算 v1 raw robust score `R_c=-|x-m_c|/sigma_c`。只在 98 个 supported classes 内按 query 做 classwise centering/scaling：`L_c=(R_c-mean_S(R))/max(std_S(R),1e-12)`。对每个 unsupported class 精确指定 `L_c=0`。因此零代表 neutral local contribution，而不是伪模板或负无穷。对 local-ineligible query，全部 `L_c=0`。

全局 logits 仍在每个 query 的 102 类内标准化为 `G_c=(logit_c-mean(logit))/max(std(logit),1e-12)`。v1 的 convex form 被本 v2 的显式 additive neutral form替换：

`F_c = G_c + alpha * L_c`。

这保证 unsupported classes 的局部贡献严格为零，其分数保留冻结 global standardized score；它们仍与受到正/负 local contribution 的 supported classes 公平竞争。alpha grid 仍为 `[0,.05,.10,.20,.30,.50,.75,1.00]`，只用 official Day0 valid 的全 102 类 top-1 accuracy 选择，完全并列取较小 alpha。unsupported valid 样本的损害完整进入选择指标，不使用 class-specific alpha、threshold、补偿或 future 信息。

local rank 使用完整 102 维 `L` 与 class-id tie break；unsupported true class 的 local score 是明确的 neutral 0，而不是 missing template。trace 不满足 500 budget 时 local rank 仍记为缺失，因为该 trace 没有任何局部观察证据。所有全局与融合 accuracy 始终包含全部样本。

## 预注册分层与 unsupported 风险检查

除 v1 的 overall/date/site/actual confusion-pair 汇总外，主统计固定增加以下完全由 source support status 定义的 strata：

1. `true_template_supported` 与 `true_template_unsupported`；
2. `global_top1_supported` 与 `global_top1_unsupported`。

`local_rank_diagnostic.csv` 报上述 strata 的 coverage/rank/top-k/margin；`fusion_repair_table.csv` 报上述 strata 的 transition counts、Net Repair 和 accuracy。`per_site_date_breakdown.csv` 保留全部 102 类，因此四个 unsupported true sites 明确可见。另在 `summary.json` 与 `RESULTS.md` 明确报告 true-unsupported 的 correct→wrong 数及其相对 global-correct 数；所有这类损害完整计入主 all-class Net Repair，不作豁免。

## 保守 label-free sensitivity

固定增加一个非主规则 sensitivity：若冻结 global top-1 属于 `U`，则对该 query 令 sensitivity prediction 等于 global top-1；否则使用同一个 Day0-selected primary alpha 和 additive fusion。该 gate 只查看冻结 global prediction 与 source-defined `U`，不查看 query 标签、future batch 或 margin。它记为 `run_mw_50_99__bypass_global_unsupported`，在 `fusion_repair_table.csv` 和 `specificity_controls.csv` 单独报告；它不替代 primary，不参与 primary verdict，也不能反向改变 alpha。

## 裁决与权限

v1 的三个允许裁决、日期/网站/rank/specificity gates 全部保持；裁决只使用未经 bypass 的 primary all-class 结果。unsupported strata 和 bypass sensitivity 只解释风险与稳健性。未来标签仍只在全部 templates、scales、support set、alpha 和 logits/predictions 固定后用于 post-hoc development metric/attribution。训练、adaptation、selector、learned gate、TTA、window search 与后续 Job 仍禁止。

本文件的 SHA-256 在 future scoring 前写入 `plan_freeze_v2.json`；该记录同时确认当时尚无 `frozen_day0_parameters.json` 或任何本 run future logits。
