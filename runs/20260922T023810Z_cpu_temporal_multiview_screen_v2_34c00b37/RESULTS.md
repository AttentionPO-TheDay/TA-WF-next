# 实验结果

## TemporalDrift CPU 多视角筛查完成

2026-09-22。v1在训练前因source/valid方向重复STOP；v2从source排除168条命中完整official valid方向哈希的行，并在source内去除718条重复行，剩余18553条canonical候选后按类抽样。最终source2040、valid510、五日期各2040；选中角色间方向哈希交叉为0。

四视角各训练三个seed，表中为accuracy均值±样本标准差（%）：

| 角色 | packet | windows | exact-run | coarse-run |
|---|---:|---:|---:|---:|
| valid | 23.53±0.90 | 27.84±0.85 | 10.59±0.52 | 10.20±0.71 |
| Day14 | 22.25±1.12 | 25.70±0.19 | 10.03±1.21 | 9.67±0.75 |
| Day30 | 21.60±0.76 | 24.36±1.01 | 9.41±0.95 | 8.97±1.74 |
| Day90 | 18.24±0.83 | 20.96±0.88 | 9.28±0.63 | 9.44±0.23 |
| Day150 | 15.11±0.81 | 17.91±0.45 | 8.12±0.48 | 7.39±0.35 |
| Day270 | 13.94±0.99 | 15.39±0.66 | 7.63±0.79 | 7.32±0.38 |

## 预注册判读

windows相对packet的三seed平均accuracy差为Day14 +3.45、Day30 +2.76、Day90 +2.73、Day150 +2.79、Day270 +1.45 pp；每个日期的三个seed均为正，共15/15正配对。预注册门槛要求五日期至少四日均值为正，windows以5/5通过，列为后续候选正向。

exact和coarse在五日期均低于packet，分别0/5通过。它们仍高于随机水平且随日期下降较缓，但源valid辨别力很低，存在明显floor effect，不能解释为更抗漂移。

## 能支持与不能支持的结论

本轮首次在真正的TemporalDrift日期上得到windows的跨seed正向开发证据：使用当前轻量模型时，windows在所有日期取得更高绝对accuracy和macro-F1。它支持继续研究window表示及其与packet的组合。

它尚未证明“减少时间漂移影响”。从各自valid到Day270，packet平均下降9.59 pp，windows下降12.45 pp；windows的绝对优势随时间从Day14 +3.45 pp缩小到Day270 +1.45 pp。模型结构与参数量不同，差异可能来自归纳偏置、优化或表示；不能单独归因于生成器。所有window模型最佳epoch均为15，训练充分性仍有限。

run结果说明当前512-token保序卷积读出不足以竞争packet/window，不等于run生成器无信息；但在下一轮融合前不应把run作为主候选。

## 完整性

独立核验通过：72行指标、128520条预测、方向交叉0、封存哈希和valid选模0错误。12个模型全部完成；执行2分13.51秒，峰值RSS3318808 KiB，GPU隐藏。完整逐seed、macro-F1、相对valid下降及判读见`artifacts/metrics.csv`和`artifacts/date_summary.json`。

下一步应做容量受控的packet+window增量实验：相同packet主干与分类头下比较真实window分支及匹配控制，检验windows是否为packet模型提供额外可用信息，而不是仅因当前两种模型架构不同。
