# P0 重复与 split 泄漏审计

- Shusterman HDF5：142 文件，规范化行 hash 556926 条
- Proteus-family NPZ：0 文件，行 hash 0 条
- HDF5 重复计数：{'same_date_train_test': 2, 'conflicting_site_rows': 38, 'within_file': 11, 'cross_date': 25, 'cross_component': 3}
- NPZ 重复计数：{}
- 读取错误：HDF5 0；NPZ 30
- 全程 CPU 只读；未训练、未适配、未访问外部未来测试。

## 裁决边界

- 只有明确列出的重复才进入 P0 处理候选；不自动删除或改写共享数据。
- 重复向量的标签冲突需要按 occurrence 审核；在隔离完成前不建立联合训练 manifest。
- 该 run 不创建新的 split，也不根据 query 标签选模。

完整 occurrence 记录见 `artifacts/p0_audit.json`。
