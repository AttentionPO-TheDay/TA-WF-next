# 来源与复用

- 模型实现从 `runs/20261004T011930Z_local_time_stretch_aug_302fcd01` 最小复制，保持 wide progressive generator、BatchNorm、Transformer、AdamW、span mask 和评估流程；本 run 的 `freeze.json` 固定本地复制版本。
- `artifacts/prepared.pt` 是同一 TemporalDrift source/valid 准备产物的只读复制，未重新划分数据。
- 新增 `temporal_order_mask` 仅在本 run 的 `worker.py` 中实现，不依赖旧目录运行时导入；不读取标签或未来数据。
