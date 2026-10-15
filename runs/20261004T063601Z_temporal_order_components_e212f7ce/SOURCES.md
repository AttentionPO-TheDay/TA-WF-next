# 来源与复用

- 模型实现从 `runs/20261004T044803Z_temporal_order_mask_3f751089` 最小复制，保持 wide progressive generator、BatchNorm、Transformer、AdamW、span mask 和评估流程；本 run 的 freeze.json 固定本地复制版本。
- `artifacts/prepared.pt` 是相同 TemporalDrift source/valid 准备产物的只读复制，未重新划分数据。
- 新增四种顺序掩码组件仅在本 run 的 worker.py 中实现，不依赖旧目录运行时导入，不读取标签或未来数据。
