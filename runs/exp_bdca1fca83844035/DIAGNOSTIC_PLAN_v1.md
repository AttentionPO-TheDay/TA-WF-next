# Burst-Level Temporal Drift Characterization — Diagnostic Plan v1

冻结时间：2026-09-18（Asia/Shanghai）。本计划在读取未来日期属性结果前写入；后续实现错误只能保留修订记录后修复，不得按结果改变主定义。

## 权限与研究边界

本实验是 TemporalDrift 上的 descriptive development diagnostic，不是外部确认、模型评价或已成立的方法贡献。允许在统计汇总阶段使用未来真实网站标签比较同网站与不同网站；这些标签不得拟合部署权重、选择稳定位置、调阈值或定义未来可用规则。训练、微调、backbone 解冻、表示学习、分类器、TTA、few-shot/current-label adaptation、Packet/Burst CNN 比较和 GPU 运行均禁止。

只读输入为 `/mnt/data2/ren/datasets/TemporalDrift/{train,day14,day30,day90,day150,day270}.npz`、正式 v3 split `runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json` 及其 source length/content audits。`valid.npz` 是 source checkpoint selection 角色，不并入 Day0，以免重复正式 manifest 已排除的 train-valid admitted-input 交集。`tam_*.npz` 不使用。

## 实际 schema、日期和统一观察边界

NPZ 只有二维 `X` 与一维 `y`。`X` 为 `float64 [N,10000]` 有符号相对 packet timestamp，符号是方向、绝对值是相对时间；不是 packet size。`y` 为可整数化的 `float64 [N]`，共 102 类。行数：train 19,439；Day14 22,603；Day30 22,867；Day90 28,599；Day150 24,064；Day270 19,935。

现行正式输入预算固定为前 `L=5000` 个原始位置，再逐元素 `sign`；正、负、零分别为 `+1/-1/0`。所有 direction-only packet 与 burst 统计严格使用同一个 `X[:L]` 前缀，不因未来结果改变 L。有效原始长度定义为前 L 内最后一个非零有限位置加一；其后的连续零是输入 padding。首包零时间戳在理论上可能与 padding 混淆，因此审计会分别报告 leading/interior zero；只有经逐行验证的 trailing zero suffix 才当 padding。任何 nonfinite 或 interior zero 都触发保守异常记录，不跨越异常位置拼接 run。

Day0 样本固定为 v3 manifest 三个 canonical source role 的并集（18,553 行）；该 manifest 已先排除与 official valid 重叠的 admitted input，再按 `SHA-256(int8 sign(X[:5000]))` 去重。未来日期没有正式 split；为防重复行改变经验分布，本实验在看属性结果前固定采用同一 admitted-input hash 的 date-internal canonicalization，保留最小 row index并审计跨标签 hash 冲突。原始与 canonical 数量都报告。

## Burst 与边界定义

burst 是有效方向序列中连续相同 `+1` 或 `-1` 的 maximal run。signed burst-size sequence 保存每个 run 的精确方向与精确整数长度；它是方向序列的可逆 RLE，不是过滤或信息丢弃。对每条 canonical trace 执行 decode(encode(direction)) 的逐元素 exact round-trip，另检查 run 非零、相邻符号交替、长度和等于有效长度；任何失败即停止解释。

若原始有效长度大于 L 且 `sign(X[L-1]) == sign(X[L]) != 0`，前缀最后一个 run 标为 terminal/right-censored；若 L 恰逢方向转换则不标记；有效长度不超过 L 的自然末段不称为 observation-boundary censoring。主分析保留所有已观察 run 并携带标记；固定敏感性会排除每条 trace 的 terminal run。

## Direction-only 主属性

每 trace 预先计算：有效长度、padding fraction、burst count、transition density `(burst_count-1)/(valid_length-1)`、outgoing packet mass fraction；全部 run、incoming run、outgoing run的 mean/median/p90/max length；incoming/outgoing mean-length log ratio `log1p(mean_out)-log1p(mean_in)`；相邻 run `log1p(length)` 的 Pearson correlation（少于 3 run 记缺失）；相邻 outgoing/incoming run 的 log-length ratio均值。方向必然交替，不把 direction pattern 当独立特征。

网站×日期中心采用 trace-level 属性的 median。Day0 内部噪声对每网站、每属性用 seed 20260918 的 200 次不重叠随机分半，记录两半 median 的绝对差分布。Day0→future temporal shift 为 future median 与完整 Day0 median 的绝对差；报告 raw shift、Day0 split median/q95、`shift / max(split_median, 1e-9)` 及是否超过 Day0 q95。主要汇总单位是 102 网站 × 5 future dates，不以 p 值代替 effect size。

同属性判别性在每个日期对 102 个网站中心做全部异网站绝对距离；报告其中位数，并与同日期同网站 temporal shift 中位数形成 `different_site_distance / max(same_site_shift,1e-9)` separability。属性成为 follow-up 候选须同时满足：至少 80% 网站×未来日期不超过各自 Day0 q95、五个日期中至少四个的异站/同站中位距离比不小于 2、且下述边界/长度敏感性方向一致。该门槛只决定本诊断裁决，不产生部署 selector。

## 两套位置坐标

A. 固定 raw packet-index：在统一 L 上预先固定 10 个等宽 500-packet windows。仅对窗口内真实有效 packet 计算 outgoing fraction、transition density、起始于窗口内的 run length mean/median、覆盖窗口的 run length mass-weighted mean，并单独报告窗口可观察率；固定窗口 packet count 不作为发现。

B. 相对 burst-order：按 `floor(run_index * K / burst_count)` 分箱，主 K=10；报告每箱占该 trace 有效 packet 的 mass fraction、outgoing packet mass fraction、run-length mean/median及方向条件 mean length。每箱 burst count 不作为结果。相同 burst 百分位不等于相同网页加载阶段，不作广告、JS、异步资源或页面阶段归因。

位置指标沿用相同 Day0 split baseline 与同站 temporal shift定义。bin 数 5 和 20 仅作固定敏感性，不从中挑有利设置。

## 边界、长度、样本量与 timing 敏感性

固定检查：(1) 全 canonical 主集；(2) 排除 terminal/right-censored run 后重算 run 属性；(3) 仅有效长度达到完整 L；(4) 对有效长度至少 500 的 trace 使用共同前 500 packet预算；(5) 依据 Day0 有效长度四分位边界分层汇总；(6) 相对 burst bins K=5/10/20。逐日期×网站样本数完整报告，网站中心避免大类直接支配总体，Day0 分半和异站比较均在网站层完成。

timing 只在独立审计确认：非零 `abs(timestamp)` 有限且逐 packet 非降、符号解释与正式文档一致时执行。burst duration=`last_abs_time-first_abs_time`，inter-burst gap=`next_first-last_current`。单包或重复 timestamp 的 duration=0；不构造无穷 rate。可选 size/duration rate仅对 duration>0计算，并报告 duration<=0 的 burst/trace比例；timing 结果不得归因给 burst-size RLE。

## 裁决

- `SUPPORTS_FOLLOWUP`：至少一个预注册结构属性/位置规律同时满足稳定性、判别性和全部关键边界/长度稳健性门槛，并能提出只依赖“历史有标签 + 当前无标签批次”的后续可检验假设。
- `INCONCLUSIVE`：有局部规律，但稳定性、判别性或稳健性不足。
- `STOP_THIS_BURST_ROUTE`：稳定属性无判别性，或位置规律主要由长度/padding/截断/binning解释，或只在少数网站/日期成立，或利用必须依赖当前真实标签。

裁决只回答是否值得继续 burst 结构机制研究，不替 Host 作路线决定，不自动创建训练 Job。
