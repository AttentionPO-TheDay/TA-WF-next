# 来源与复用

- 模型实现从 `runs/20261003T141442Z_temporal_drift_input_aug_896bdcb1` 最小复制，保持 wide progressive generator、BatchNorm、Transformer、AdamW、span mask 和评估流程；本 run 的 `freeze.json` 固定本地复制版本。
- `artifacts/prepared.pt` 是同一 TemporalDrift source/valid 准备产物的只读复制，未重新划分数据。
- 新增 `local_time_stretch` 仅在本 run 的 `worker.py` 中实现，不依赖旧目录运行时导入。该实现用线性插值和通道总量重标定，不读标签或未来数据。
