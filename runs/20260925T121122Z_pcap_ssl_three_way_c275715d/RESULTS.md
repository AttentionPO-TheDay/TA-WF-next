# 实验结果

状态：completed（v2 修订后复测）。

## 结论

本轮没有得到外部 PCAP 无标签预训练改善少标签分类的正向证据。三条件使用同一 219,494 参数 packet+window 融合模型、同一 5-shot/class TemporalDrift 训练清单和同一 valid 选模规则；`scratch_fusion` 的 valid macro-F1 均值为 20.298%，`pcap_packet_pretrain` 为 19.082%（−1.216 pp），`pcap_window_pretrain` 为 19.186%（−1.112 pp）。逐 seed 的 valid 差如下：

| 条件 | seed 1729 | seed 3407 | seed 2026 | 均值 |
|---|---:|---:|---:|---:|
| scratch_fusion | 20.950 | 19.817 | 20.127 | 20.298 |
| PCAP packet 预训练 | 18.992 | 18.998 | 19.255 | 19.082 |
| PCAP window 预训练 | 19.160 | 18.303 | 20.096 | 19.186 |

packet 预训练相对 scratch 的 valid 差为 −1.959/−0.818/−0.872 pp；window 预训练为 −1.791/−1.514/−0.032 pp，三个 seed 均未超过 scratch。五个开发日期的均值差（Day14/30/90/150/270）分别为 packet +0.280/+0.131/−0.295/+0.262/−0.185 pp，window −0.371/−0.331/−0.184/+0.025/−0.254 pp；均未满足预定门槛。

## 方法与数据

PCAP 目录只读解析，140 个文件中抽取 465 条主 flow 方向序列；预训练不读取文件名类别。`pcap_packet_pretrain` 和 `pcap_window_pretrain` 在 PCAP 上各自进行 8 epochs、15% 输入遮挡重建，再用 TemporalDrift 每类 5 条（共 510 条）有标签样本微调 20 epochs。window 统计现已直接调用本项目 `traffic_views.py`，宽度为 50/250。TemporalDrift valid 仅用于选最早最佳 epoch；五日期仅作选模后的开发诊断；WTT/AWF 和预留 future 未访问。

## 限制与解释

这不是对无标签预训练路线的普遍否定。PCAP 是 VPN/非 VPN 应用流量，不是网站指纹流量；465 条主 flow 对 Transformer/ET-BERT 级预训练很小；遮挡重建只是最小自监督目标，不是 ET-BERT 的完整 MBM+SBP；首包方向也不保证等于客户端方向。更合理的解释是：在当前小语料、跨域分布和目标下，未观察到迁移收益，且预训练初始化可能损害少标签优化。不能据此否定更大、同域的无标签网站流量或更合适的预训练目标。

首轮结果因输出路径错误未完整封存，随后又发现自写 window 统计与正式生成器边界值不一致；首轮产物保留但无效。v2 仅修正这两个实现问题，未改变数据、标签预算、模型、seed 或选模规则。

## 核验

`artifacts/integrity_v2.json`：63 个预测集、checkpoint 重载和指标重算 0 错误；source/valid 方向重叠 0；PCAP 序列与 source/valid 方向完全重叠 0；每类少标签数均为 5；三条件参数量均为 219,494。原始 PCAP 和 TemporalDrift 数据未修改。
