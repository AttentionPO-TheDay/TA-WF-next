# 多因素可评价子集与身份门槛审计 v1

问题：Version/Network/Behavior 的哪些行能以已有逐行证据归因，并与纯净源权重的 18,553 条训练输入精确隔离？仍缺什么才可进入性能评分？

输入：`configs/datasets.json` 定位共享只读数据；P0 的 `core_file_index.json` 与 `core_row_group_ids.npz` 提供完整 10000 维原始 trace 精确重复组；联合审计的 `conservative_train_candidates.csv` 给出源域训练行；Version 身份审计的四个 `version_*_identity.csv` 给出逐行锚点归因；Behavior 原始 `subpage.npz` 仅读取 `url.npy` 和 `y.npy` 成员以核对 URL 重复组。审计使用标签核对一致性与覆盖，不训练、不预测、不选模。旧审计结果只作为来源，不冒称新采集证据。

输出每个目标文件的逐行候选清单：原行号、数值标签、原始精确重复组、源训练重叠、版本归因/锚点重用、或 Behavior 精确 URL 组与重复数。另输出各切片全行与候选计数、标签覆盖、跨目标重用、来源散列和门槛判定。Version 的未知 10,017 行保留为未归因；唯一版本归因若来自 train/valid 锚点，明确标为锚点重用，不称独立新采集。Network 只称地理文件，不推断丢包/延迟。Behavior 精确 URL 不是 homepage 网站身份映射。

本轮不生成随机 reference/query 划分，不开展目标评分、适应或权重更新；不将观察过的行宣称确认集。进入后续评分需在本轮清单上另外冻结官方或可信的站点身份、切片语义、输入兼容和无标签 reference/query 规则。预算 CPU 2 线程、墙钟 3,600 秒、输出≤0.25 GiB；输入校验或计数失败即保留失败证据并停止。

修订记录（正式完成前）：首轮执行在读取 Network 标签后的输入成员哈希登记处因把工作区外共享路径误作工作区相对路径而失败；`artifacts/failure.json` 保留。只修正该路径键的记录方式，同一协议重新运行；失败前无预测、评分或方法反馈。
