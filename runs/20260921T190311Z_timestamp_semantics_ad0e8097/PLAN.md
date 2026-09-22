# 20260921T190311Z_timestamp_semantics_ad0e8097

问题：查明Proteus时间回退的数值特征、方向关系和处理语义；不改数据不训练

状态：frozen audit-only，2026-09-22。只读时间语义诊断。

读取STATUS/PROTOCOL/HANDOFF及官方固定提交处理代码。范围：Version四目录train/valid及048/drift（覆盖版本语料，包括重复视图但不重复扫描另三drift）；Network全部7文件；Behavior仅subpage（其余与Network相同），共17文件。源路径来自configs/datasets.json。只读X，不读取y/URL，不打开其他数据集。

单CPU memmap顺序扫描。统计有效非零段相邻abs时间差、同/异方向回退、各方向子序列回退、回退幅度、全局时间前沿回退、padding/非有限值、run duration。每文件最多128条固定等距样本进行排序敏感性/四舍五入诊断，不修改或生成替代数据。保存可复查数值例子但不解释为因果证明。读官方处理代码；若无原始采集/转换链，不猜测真实根因，明确待补证。

无训练/适应/模型评分/选参/新split，数据只读，输出本run。完成完整性检查后停止。不默认排序或截负值作为修复。
