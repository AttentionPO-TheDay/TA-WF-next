# 来源与复用

- 模型实现从 `runs/20261003T131656Z_label_smoothing_6d748296` 最小复制，保持 wide progressive generator、BatchNorm、Transformer、AdamW、span mask 和评估流程；本 run 的 `freeze.json` 固定本地复制版本。
- `artifacts/prepared.pt` 是上一轮同一 TemporalDrift source/valid 准备产物的只读复制，未重新划分数据。
- 新增 `temporal_origin_jitter` 仅在本 run 的 `worker.py` 中实现，不依赖旧目录的运行时导入。
