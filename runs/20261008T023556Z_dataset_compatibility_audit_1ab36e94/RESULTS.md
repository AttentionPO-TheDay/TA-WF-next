# 数据集兼容性审计结果

- 完成时间（UTC）：2026-10-08 02:49:09
- 候选目录：10；缺失/错误：0
- 设备：CPU；训练、TTA、外部未来评分：均未执行。

## 初步筛选

- 核心 Proteus-family NPZ 候选：TemporalDrift、VersionDrift、NetworkDrift、BehaviorDrift；需先验证 X 的时间戳语义和跨目录标签体系。
- AWF 只能通过方向子表示接入，不能和 timestamp 数值直接拼接。
- WTT-Time 与 Shusterman 可作为有符号包大小候选，但需独立前端/分类头和 HDF5 内部 schema 审计。
- Defense、OpenWorld 保留为独立任务评价；PCAP 暂不进入第一版监督联合训练。

## 限制

- 本轮使用标准库读取容器和 NPZ header；未加载数组到模型。
- HDF5 内部 dataset/label 结构尚未解析；尚未声称跨数据集网站标签相同。
- 小文件 hash 只用于边界提示，未完成样本级重复检测。

详细产物见 `dataset_inventory.json`、`compatibility_report.json`、`duplicate_boundary.json`。
