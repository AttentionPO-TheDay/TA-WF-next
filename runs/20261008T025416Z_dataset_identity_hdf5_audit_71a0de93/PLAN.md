# 20261008T025416Z_dataset_identity_hdf5_audit_71a0de93

问题：在第一轮兼容性筛选后，解析 Shusterman HDF5 内部字段，核对候选数据集的标签集合/环境元数据，并审计可行范围内的精确向量重复关系。

范围：CPU 只读；训练、微调、TTA、模型前向、目标标签选模、外部未来评分全部关闭。读取标签仅用于结构、身份映射边界和重复冲突审计，不用于性能或方法选择。

输入：共享只读 `/mnt/data2/ren/datasets` 下 Shusterman 142 个 HDF5、Temporal/Version/Network/Behavior/Defense/OpenWorld/AWF NPZ 标签和 schema；WTT-Time manifest/label map 仅作现有协议边界引用，不重建训练样本。

检查：
- HDF5 dataset keys、shape、dtype、site/date/tag/version 元数据；
- all-zero、suffix cap、非整数/非有限值、内部 zero；
- Shusterman 全文件精确向量 hash，区分同文件/同日期/跨组件/跨日期重复及标签冲突；
- NPZ label key、dtype、shape、原始标签集合与每文件计数；
- NPZ 仅做每文件固定首尾/均匀抽样行 hash，不宣称全量跨容器重复已排除；
- 原始标签值的交集只作数值碰撞提示，不能据此认定跨数据集网站身份相同。

停止条件：所有可读文件完成审计；单文件读取异常保留错误并停止该文件，不修改原数据。
