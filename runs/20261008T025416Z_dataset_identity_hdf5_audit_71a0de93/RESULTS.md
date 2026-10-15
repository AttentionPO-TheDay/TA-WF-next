# 第二轮数据身份与重复审计结果

- HDF5 文件：142；规范化行 hash：556926
- HDF5 重复计数：{'same_date_train_test': 2, 'conflicting_site_rows': 38, 'within_file': 11, 'cross_date': 25, 'cross_component': 9903}
- NPZ 数据集：7；AWF/小文件执行确定性抽样 hash。
- 未训练、未适配、未访问外部未来测试。

## 解释边界

- HDF5 内部字段和 Shusterman 的网站/日期/浏览器/网络元数据已解析。
- 原始标签集合交集只表示数值/字符串碰撞提示，不能证明不同数据集的网站身份相同。
- NPZ 大文件未做全量样本 hash；抽样相同不等于确认重复，未发现也不等于证明无重复。
- 只有通过人工或来源 manifest 建立跨数据集身份映射后，才能训练共享分类头。

详见 `artifacts/hdf5_identity_audit.json`。
