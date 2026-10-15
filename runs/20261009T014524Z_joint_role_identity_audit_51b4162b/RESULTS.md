# 联合数据角色与标签身份审计

状态：完成（CPU 只读，2026-10-09）。本 run 复用 P0 已逐字节核验的输入分组，没有重新扫描原始 X；只读 P0 分组、标签、URL 元数据和文件 stat。未训练、未 TTA、未做性能评分，未改动原始数据、原 split 或 P0 候选。

## P0 产物的独立重建

- 三种输入视图（完整 10,000、前 5,000 float32、前 5,000 方向 int8）的全行分区是同一分区，逐行双射核验通过；不是只比较汇总数字。
- 三个视图各自 490,842 个重复成员逐行回查通过，group id、标签、原角色、action/reason 与 P0 CSV 一致。
- 核心候选 114,307 行和 Shusterman 候选 25,943 行均逐行重建一致，所有 `applied=False`；P0 冻结 hash 和核心/HDF5 源文件 stat 均保持不变。

## 保守联合训练候选

核心 30 个 NPZ 的原始 train 共 132,860 行。保护所有原非 train 角色并按原文件/行顺序保留 train 组首行后，得到 18,553 个唯一输入候选，全部来自 TemporalDrift/train；其余 114,307 行排除：113,589 行匹配原非 train 角色，718 行为 train 内副本。候选按 102 个数值标签分布保存于 `artifacts/conservative_train_candidates.csv`，仍需新的数据角色、标签映射和训练协议批准才能使用。

这解释了当前联合池的边界：VersionDrift 的四个 train、NetworkDrift train、BehaviorDrift train 都与已保护角色共享输入，因此在保守规则下没有可直接加入的独立 train 行。该结果不否定它们作为各自条件的评价/诊断角色。

## 跨家族标签对应证据

对共享精确输入的唯一 group 统计如下：TemporalDrift–VersionDrift 28,599；TemporalDrift–NetworkDrift 19,935；TemporalDrift–BehaviorDrift 19,935；NetworkDrift–BehaviorDrift 19,935；VersionDrift 与 Network/Behavior 没有共享 group。已有共享 group 覆盖 102 个标签，数值标签对全部相等；这支持“已观察共享内容中的标签对应”，不能单独证明未匹配样本、采集会话或官方网站身份映射。

逐对证据在 `artifacts/cross_family_label_evidence.csv`。数值相等没有被当作跨数据集身份先验；Version 的 10017 条未匹配 drift 仍不能恢复身份。

## Behavior subpage URL

20,400 行中有 17,739 个精确 URL，2,661 个额外副本；精确 URL 没有跨标签冲突。去小写并去掉 `www.` 仅用于描述时，仍有多个 hostname 的标签（页面路径/子域造成），没有 hostname 同时对应多个标签。该字段只能支持 subpage 内的页面标签检查，不能证明 homepage 对应关系或访问独立性。详情见 `artifacts/subpage_url_identity.json` 和 `subpage_label_hostnames.csv`。

## 结论与下一门槛

当前可以形成一个“候选清单”，不能直接形成已批准的混合训练数据集。下一步需要先冻结：跨家族标签映射、哪些原评价角色可以转为训练、Behavior 页面级隔离规则，以及 TemporalDrift 开发与 WTT/AWF 外部评价的权限边界。若将 18,553 候选用于新训练，必须另立实验版本并保留本次结构审计暴露记录。

独立核验结果见 `artifacts/audit.json`；哈希和产物清单见 `artifacts/integrity.json`。
