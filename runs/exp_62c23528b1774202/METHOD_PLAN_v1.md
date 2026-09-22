# Equal-budget region relevance method plan v1

冻结日期：2026-09-17（Asia/Shanghai）。本文件在生成本实验任何 B/C/D/消融 query 预测或读取其 query truth/指标前冻结；结果出来后不改写。本实验使用已观察的 TemporalDrift 开发数据，不是外部确认。

## 问题、范围与权威输入

唯一问题是：相同历史存储、相同正式每类 3-shot 当前标签和相同冻结 backbone 下，区域级历史保留与冲突抑制是否比普通多原型和网站级整体混合更有效，并超过现有最强 support-CV 3-shot。

- backbone 仅为 DF epoch 29（checkpoint SHA-256 `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`）与 VarCNNDirection epoch 23（`fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83`）。表示是冻结的 512 维、逐行 L2 归一化 global embedding。
- 日期仅 Day14/Day90/Day270；support seeds 仅 1729/6238/20260916；类别 102；正式输入严格为既有嵌套 manifest 的每类前三条，即 306 条。额外七条不得参与区域、阈值、权重、冲突、选择、预测或任何算法计算。
- 只读复用 `exp_04faf4088b604155` 已验证的 source/current embedding cache、source geometry 和 frozen support order；只读复用 `exp_376fca9354214097` 的 3-shot A 与 common query。query 固定为其 canonical pool 减去预先存在的完整 10-shot manifest，故额外七条也不作为本轮 query；这是复用既有 A 所必需的共同评分集合，不赋予算法使用这七条的权限。
- 新 backbone 训练、微调、optimizer/backward、checkpoint 写入、表示更新、TTA、外部数据、新 backbone、训练 seed 扩展及超参搜索均为 0。

## 冻结历史库与严格预算

每个 backbone 的唯一历史候选库是 `exp_04faf4088b604155` source geometry 中每类 3 个 source-only farthest-first exemplar。第一个 exemplar 是离 source 类中心最近的 source row；其后依次选择到已选集合最大相似度最小者，完全并列取最小 source row id。该库共 102×3=306 个 float32 512 维 exemplar，B/C/D/消融逐字节读取同一个只读 NPZ，不复制、增删或重选。

共享历史包还包含 306 个 int64 source row id 和 102 个 float32 source-only q95 阈值。历史数值 payload 为 `306*512*4 + 306*8 + 102*4 = 629,544` bytes；以实际生成的共享 NPZ 文件大小另行报告。阈值即使 B 不使用也保留在其完全相同输入包中。算法产生的 support-conditioned 权重不是额外历史信息；为核算运行态，B/C/D/消融都物化同形 `(102,3)` float32 weight array（1,224 bytes）。当前 3-shot embedding/current prototype 属于共同当前预算，所有 B/C/D/消融相同。区域方法不得访问任何额外 source row、原型、标签或索引。

A 是已冻结的强常规方法，其 source 线性头/候选族表示与 exemplar 库不同，故另报其实际 artifact 字节，不伪称与 exemplar 表示逐字节同构；它不访问本轮额外历史候选。等历史库的机制归因以 B/C/D/消融的完全相同包为严格比较。

## 确定性区域与冲突证据

每个 `(class, exemplar-slot)` 就是一个历史区域，不重新聚类。对配置的正式 3-shot：

1. `own_sim[c,r]` 是历史 exemplar `h[c,r]` 对同类三条 current support 的最大 cosine similarity。
2. `alien_sim[c,r]` 是它对其余 101 类共 303 条 current support 的最大 cosine similarity；`alien_class[c,r]` 是最大者标签，完全并列取最小标签、再取最小 row id。
3. 区域冲突严格定义为 `alien_sim > own_sim` 且 `1-alien_sim <= tau[alien_class]`，其中 `tau` 是该 backbone 的 source-only q95 阈值。冲突权重 `w[c,r]=0`，否则 `w[c,r]=1`。不设 margin、不搜索阈值、不按 query 调整。

该规则只使用冻结 source 历史包和正式 3-shot。默认保留 source 历史区域，只抑制已有 current 证据表明更符合另一网站且比本网站 current 证据更近的区域。

## 五个正式方法

共同 current score 为 query 与该类三条 support 均值后再 L2 归一化所得 prototype 的 cosine。共同 historical region score 为 query 与三个历史 exemplar 的 cosine。所有最终预测按 class argmax，完全并列取最小 class。

- **A — `A_support_cv_3shot`**：直接引用 `exp_376fca9354214097` 同 backbone/date/seed/shot=3 的 `support_selected_baseline` 预测及冻结 support-CV 定义；不重选候选、不换 donor。它是必须超过的现有最强同标签预算基线。
- **B — `B_plain_multiprototype`**：每类取 current prototype 与全部三个历史 exemplar cosine 的最大值。这是常规 max-cosine 多原型，等价于既有已验证 probe；无权重、无冲突抑制，不故意弱化。
- **C — `C_site_global_weight`**：先计算与 D 相同的三个 `w[c,r]`，网站全局历史权重 `q[c]=mean_r w[c,r]`。该类历史分数为 `-1 + q[c]*(max_r(sim_hist[c,r])+1)`，再与 current score 取最大。每网站仅一个平滑全局权重；`q=1` 等于 B，`q=0` 禁用该网站全部历史。
- **D — `D_region_conflict_suppression`**：每个区域历史分数为 `-1 + w[c,r]*(sim_hist[c,r]+1)`，再与 current score共同取最大。只有满足冻结冲突条件的具体区域被禁用。
- **D-ablation — `D_ablation_uniform_site_retention`**：使用完全相同三个冲突位与历史包，但塌缩为网站级统一二元决策：仅当三个区域均非冲突时统一保留全部历史，否则统一禁用该网站全部历史；然后与 current score 取最大。该消融不引入新信号或预算。

C 是尽可能强的同信号平滑网站级整体加权；D-ablation 是关键“去区域”统一保留消融。两者都与 D 的唯一区别收缩在决策粒度/塌缩方式。

## Query-label firewall 与执行顺序

1. `preflight`：哈希并核验全部计划、cache、geometry、frozen support、A evaluation 与 donor integrity；不加载新 query truth或指标。
2. `freeze`：从 source geometry 建立一次共享历史包；只读取 current embedding、`support_order[:,:3]` 与 query row id，生成 B/C/D/消融预测、冲突位和预算记录。不得读取 current NPZ `y`，不得读取 A JSON 的 truth/metrics/predictions。
3. `seal`：覆盖计划、代码、共享历史包和全部未评分 frozen unit 的 SHA-256；seal 后禁止覆盖。
4. `score`：先复核 seal，随后才从已冻结 A evaluation 读取共同 query truth与 A prediction，核对 query row完全相同并评分。query 标签仅用于 macro-F1/accuracy、逐类指标及预注册差值/裁决。
5. `verify`：独立重算指标、预算和哈希。任何规则/预测变化必须另建版本，不覆盖 v1。

研究者已看过前序 A 与普通 probe 的 aggregate 结果；本 firewall 不虚构历史盲态，只保证本实验新 C/D/消融定义与预测不受其新 query 评分驱动。

## 指标、汇总与冻结裁决

主指标为 protocol 一致的 macro-F1，同时报告 accuracy。machine-readable 明细单位为 backbone×date×seed×method，共 18×5=90 行。差值逐相同 query 配对，以百分点报告。

“实质相当”冻结为绝对 seed-paired mean macro-F1 差 `<1.0 pp`。后期单元是两个 backbone×Day90/Day270×3 seeds，共 12 个。

只有以下全部满足才写“初步机制信号成立”：

1. D 相对 A、B、C 的后期 mean macro-F1 均至少 `+1.0 pp`，且对每个对照都至少 9/12 后期单元为正；
2. 对 A/B/C 每一个，DF 与 VarCNNDirection 各自合并 Day90/270 后的 D 差值都为正，且 Day90、Day270 各自跨 backbone/seed 也为正；
3. Day14 每个 backbone 的 D 相对 A seed-mean 损伤不超过 `1.0 pp`；
4. D 相对 D-ablation 的后期 mean 至少 `+0.5 pp` 且至少 9/12 单元为正，说明去区域后收益消失；
5. 历史条目、历史包、当前标签、query 和 backbone 完全匹配，且无 query-aware 选择。

预冻结停止条件：若 B 或 C 在两个 backbone 后期相对 D 均实质相当（每 backbone合并后绝对差 `<1.0 pp`），或 D 相对去区域消融未满足第 4 条，或优势只在单 backbone/单日期、来自额外存储/标签/query-aware 选择，则停止复杂区域模块路线。任一成功条件失败都不得包装为成立；10-shot 仅引用既有背景，不参与裁决。

完成 v1 的固定 18 配置、预算审计、完整性检查和 `RESULTS.md` 后停止，不创建后续实验，也不替 Host 决定下一研究路线。
