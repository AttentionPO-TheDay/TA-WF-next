# Versioned diagnostic plan v1

冻结时间：2026-09-17（Asia/Shanghai），早于本实验任何新 embedding、几何统计、3-of-10 预测或 query 评分。TemporalDrift 是已观察的开发数据；本实验不产生外部确认结论。

## 不可变输入与边界

- 权威 donor 是 `runs/exp_b471517a3e6f41e7`。只读复用其 `RECOVERABILITY_PLAN.md`、`support_manifests.json`、`data_isolation_audit.json`、36 个正式 evaluation JSON、`summary.json`、`metrics_long.csv`、`integrity_check.json` 和 `execution_record.md`；不修改、不覆盖、不重新生成正式 split/support/query 或原 3-shot 结果。
- checkpoint 仍为 DF epoch 29 (`1bf851...c9133ae`) 与 VarCNNDirection epoch 23 (`fc3ade...a1ff83`)；模型始终 `eval()`。输入仍为有符号时间戳前 5000 位的 `sign`，形状 `[B,1,5000]`。embedding 仍是 `forward_features` / 原 `mlp` 输入的 512 维输出，逐行 L2 归一化后用于全部诊断。
- split 仍为 `runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json` (`0f322e...f3f142`)；source 几何只使用 `supervised_train`，不使用 source validation、reference 或 source-holdout。current 仅为 Day14/Day90/Day270，support seeds 仅为 1729/6238/20260916。
- `artifacts/input_manifest_v1.json` 将在 preflight 中列出并复核全部消费文件的绝对路径、SHA-256、dataset 文件哈希、checkpoint metadata、donor evaluation 哈希和代码哈希。任一不一致即停止。
- 训练/微调、optimizer/backward、checkpoint 写入、外部数据、TTA、表示更新、复杂 adapter 和超参搜索次数均为 0。允许只读重提取 embedding，缓存只写本 run。

## Query-label firewall 与执行阶段

1. `preflight`：只核对输入与哈希，生成 `input_manifest_v1.json`；不加载 current 数组或 donor evaluation 的 query truth。
2. `extract`：冻结前向并缓存 source supervised-train 与各 current 日期全部 row 的归一化 embedding。source 标签可用；current cache 不保存标签。source-only 阈值及历史 exemplars在此阶段冻结。
3. `freeze`：只根据 manifest 中获准 support row/label、source cache 与 current embedding 计算 support-only 指标、覆盖分类及全部预测。不得读取 current NPZ 的 `y`、donor evaluation JSON 的 `query_truth`/metrics/predictions，或既有 current query 成绩。
4. `seal`：哈希全部预测、重采样、几何与指标 artifact，生成 `firewall_seal_v1.json`。seal 后规则与预测不可覆盖。
5. `score`：只有 seal 哈希复核通过后才读取 query 真标签和 donor evaluation，进行评分、错误归因、相关性及裁决。任何修改都必须新建 plan/version 和新 artifact，不得覆写 v1。

注意：研究者已从前序实验看过正式 3/10-shot aggregate 成绩；这里的 firewall 限制是禁止这些 query 反馈影响本实验新重采样、阈值、指标和规则，而不是虚构历史盲态。

## 主检验 1：有限样本估计方差

- 对每个 date×原 10-shot seed×class，确定性排列为：正式 3-shot 成员按 row id 升序置于位置 0–2，其余 7 条按 row id 升序置于位置 3–9。按字典序枚举全部 `C(10,3)=120` 个共同位置组合；同一组合用于所有 102 类。组合清单在 `resampling_manifest_v1.json` 中冻结，组合 `(0,1,2)` 精确等于正式 3-shot support。
- 所有副本只用 donor 的 `simple-maintenance`：每类三条归一化 embedding 的均值再归一化，以 query 到 102 个 prototype 的 cosine 最大值分类。无参数、无拟合、无选择。
- 为公平比较，120 个副本、正式 3-shot 和正式 10-shot全部在对应正式 10-shot query 上评分；正式 3-shot 的 donor 全 query 结果另原样引用，绝不替换。先重算 `(0,1,2)` 与 10-shot prediction 并在评分阶段核对 donor 相应 query 行上的预测完全一致。
- 每单元报告 120 个 macro-F1 的 mean/SD/min/q05/q10/median/q90/q95/max、正式 3-shot 百分位、10-shot 位置、`P(F1_3subset >= F1_10-0.02)`、site accuracy/F1 方差。best subset 只称 post-hoc oracle envelope，绝不作为方法。
- gap 描述量预先定义为：`gap=F1_10-F1_formal3_common`；`oracle_explain=clip((q95-F1_formal3_common)/gap,0,1)`（gap<=0 记 NA）；`mean_residual=F1_10-mean(F1_subsets)`；sampling spread 为 q90-q10。它们描述有限 n=3 的抽样波动/偏差，不把 oracle 当可部署结果。

## 主检验 2：覆盖不足及历史来源

所有距离均为归一化 embedding 的 cosine distance `1-cosine`。每 backbone×class 的唯一阈值是 source supervised-train 内每点到同类其他点最近距离的 95% 分位数（NumPy linear quantile）；阈值仅由 source 计算。零向量或少于两点将使 preflight/freeze 失败，不临时改规则。

对正式 3→10 的每条新增 support，计算到原 3-shot同类最近距离 `d3` 及 source 同类最近距离 `dsrc`，固定分类为：

- `near_original3`: `d3 <= threshold`；
- `far3_source_supported`: `d3 > threshold` 且 `dsrc <= threshold`；
- `far3_source_unsupported`: `d3 > threshold` 且 `dsrc > threshold`，提示 current 新区域；
- 同时保留连续距离，不只依赖二值阈值。

逐 site/config 另报告 3-shot 与 10-shot归一化中心的 cosine distance、各自平均到本 support 中心距离、7 条新增样本三类计数/比例。重复性定义预先固定：某 site 在同 backbone×date 至少 2/3 seed 出现该类新增证据；跨 backbone 重复要求两个 backbone 均满足。主结论基于距离/邻域，不使用聚类。

## 主检验 3：只有 source+3-shot 时的可识别性

对正式 3-shot及全部 3-of-10 副本逐 site计算以下四项，均不得使用新增 7 条或 query：

1. `source_coverage`: 同类 source 点中，到三条 support 任一点距离不超过该类 source-only 阈值的比例（风险方向为 `1-coverage`）。
2. `center_to_source`: 3-shot归一化中心到 source 同类归一化中心的 cosine distance（风险方向为大）。
3. `within_support_dispersion`: 三对 support cosine distance 的均值（风险方向为大）。
4. `jackknife_center_instability`: 三个 leave-one-out 二点中心到完整三点中心 cosine distance 的最大值（风险方向为大）。

事后仅检验预定义方向的 Spearman 相关、按 config 内 bottom-quartile site accuracy 定义失败的 ROC-AUC，以及指标风险四分位的表现差；不根据 query 增删或变换指标。AUC 以预定义风险方向直接计算，不翻转坏结果。

## 错误归因

在 query 解封后，对同一正式 10-shot query 比较正式 3-shot与10-shot simple-maintenance。逐 row/site记录 `repaired`（3错10对）、`harmed`（3对10错）、两者预测和真值；计算 query 到原3与新增7条真类 support的最近距离。`added_is_closer` 固定为新增7条距离严格小于原3距离；并按提供最近新增证据的那条 support在上述三分类中的类别归因。另报告 repaired 相对全部 query 的该证据富集，不以 query 选择规则。

## 唯一解释性 probe 与历史停止条件

- 不重复 `exp_a2ffb5623ad346c7` 已完成且已暴露的 shrinkage 搜索/评分；它作为既有停止条件输入单列，不能声称本实验前盲态。其正式结论是固定 source-side shrinkage 大幅缩小 shot gap但主要因为几乎不适应，未保留 Day90/270 recovery。
- 本实验唯一新增 probe 是无 query 参数的历史多原型：每类在 source supervised-train 中确定 3 个 farthest-first exemplars（第一个为离 source 类中心最近的 row，随后每次选择到已选集合最近相似度最低的 row，tie 取最小 row id），再加入 current 3-shot prototype，按每类四个 prototype 的最大 cosine 分类。只在正式原3-shot上评分，不搜索 exemplar 数、融合权重或阈值。若它解释大部分正式3→10 gap，只作为“简单历史多原型足够”的停止条件。

## 预冻结证据裁决规则

裁决单位是 18 个 backbone×date×seed 单元，并同时检查 date/backbone 异质性，不只看总体均值。

- A（估计噪声主导）证据：至少 12/18 单元 `oracle_explain>=0.5` 且至少 9/18 单元 `P(within 2pp of 10-shot)>=0.10`；同时历史支持覆盖的错误归因不满足 B。
- B（历史支持区域覆盖不足主导）证据：至少 12/18 单元中，repaired 样本的 `added_is_closer` 比例不低于全部 query 且至少高 10 pp；跨两 backbone 至少 2/3 日期中，`far3_source_supported` 在至少 2/3 seeds 重复，并且四项部署指标至少一项在两个 backbone同方向（Spearman <= -0.20 或 AUC >=0.65）预测 site/subset 失败。
- 若 A 与 B 同时满足，裁决为 C-mixed；均不满足或跨 backbone/date 不稳定为 C-insufficient。若新增远区以 `far3_source_unsupported` 为主，明确写 current-new；反之写 source-existing。这里的机器裁决是按预注册规则汇总证据，不替 Host 作后续研究立项决定。
- 停止条件：若历史多原型 probe 在两个 backbone各至少 2/3 日期解释至少 75% 正式 gap，建议停止包装复杂历史模块；若所有指标均未达到上述跨 backbone识别标准，而 post-hoc query 归因明显，则标记 query-aware、不可部署并建议停止机制开发。不得自动创建下一 Job。

## 产物与验证

计划输出：`input_manifest_v1.json`、embedding caches 与 cache metadata、`source_geometry_v1.json`、`resampling_manifest_v1.json`、未评分 prediction NPZ/metadata、`firewall_seal_v1.json`、`resampling_results.csv/json`、`per_site_stats.csv`、`support_only_metrics.csv`、`error_attribution.csv/json`、`probe_results.csv/json`、`summary.json`、独立 `integrity_check.json`、`execution_record.md` 和简洁 `RESULTS.md`。缓存和预测为诊断中间件，所有正式汇总均 machine-readable。

