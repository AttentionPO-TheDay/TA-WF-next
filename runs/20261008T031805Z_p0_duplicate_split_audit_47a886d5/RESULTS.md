# P0 重复、冲突与切分重叠审计结果

状态：完成（CPU 只读，2026-10-08）。本 run 未训练、未 TTA、未做性能评分，也没有改写原始数据或既有 split。Shusterman 的全部 142 个 HDF5 被用于结构完整性审计；其中包含 WTT 的原始未来日期，属于重复/冲突检查的数据读取，不是未来性能评价。

## Shusterman HDF5

- 扫描 556,926 行，142 个文件；全部向量通过有限、整数语义和 int32 范围检查，没有 invalid 行。
- 精确向量重复组 9,903 个，超出每组首行的重复行 9,941。重复组成员和逐行 group id 在 `artifacts/shusterman_duplicate_members.csv` 与 `artifacts/shusterman_row_group_ids.npz`。
- 31 行是全零向量，不能表示真实流量。它们形成一个全零组并带 30 个网站标签，解释了标签冲突和同日期 train/test 表象；这些行均列入 `all_zero_unusable_trace`。
- 另有 1 个非零精确向量冲突组（11 行、5 个网站标签），全部列入 `nonzero_exact_vector_conflicting_site_labels`，不得直接混合。跨组件的非零重用仍存在（动态/静态浏览器网络与辅助组件的重复），因此辅助 `static_webpages_network` 的 25,900 行全部列为 `auxiliary_component_not_in_main_pool`。
- 同日期 train/test 的精确重叠：非零为 0；仅全零组中有 2 个 train/test 成员关系。此前“同日期 2”不应作为真实流量泄漏计数。非零跨文件重叠的两对为 `2022_02_21/test`↔`start-92.../test`（990）和对应 train（8,910），需按组件角色处理。
- 逐行候选清单共 25,943 行，均保留原文件、行号、角色和 `applied=False`；没有自动应用到任何数据或 checkpoint。

## 四个核心 NPZ 与截断输入

扫描 TemporalDrift（7）、VersionDrift（12）、NetworkDrift（7）、BehaviorDrift（4）共 30 个文件、615,350 行。每行均为有限 float64，标签为有限整数；全零、非有限和前 5,000 位 float32 溢出均为 0。

对每一行同时检查三种实际输入视图：原始 10,000 位 float64 字节、前 5,000 位 float32 字节、前 5,000 位方向 int8。三种视图完全得到相同结构计数：245,500 个唯一输入、120,992 个重复组、490,842 个重复成员、369,850 个超首行副本；未发现标签冲突，也未发现仅由截断或 float32 转换新增的重复组。

逐行隔离候选共 114,307 行，全部来自 `train.npz`：

- 113,589 行与某个原始非 train 角色精确重叠，建议从未来联合训练池排除；
- 718 行是 train 内重复副本，保留每组确定性的首个 train 行作为候选，其余排除；
- TemporalDrift 原始 train↔valid 有 168 个唯一输入重叠，已全部进入上述候选；
- 三种视图的候选集合一致，候选仅是审计建议，`applied=False`，不改变现有训练/验证/测试边界。

重复成员、组关系、文件行界和候选原因分别见 `artifacts/core_*_duplicate_members.csv`、`core_duplicate_groups.json`、`core_file_index.json` 和 `core_quarantine_candidates.csv`。

## 核验与处置

`postrun_verify.py` 已独立核验：冻结文件 hash、文件/行数、三视图计数、候选行界、候选未应用标记和源文件未变更均通过（结果见 `artifacts/postrun_verification.json`）。本结果只产生可审阅的 quarantine/exclusion manifest；在完成标签/访问角色映射及预注册方法冻结前，不据此创建联合数据集或重新切分数据。

保留限制：精确输入相同不等于同一次访问已被证明；跨数据集数值标签不能直接当作网站身份映射；TAM 等进一步量化/分箱可能产生额外碰撞；原始未来日期的结构读取不构成未来性能结论。
