# 来源与边界
- 新runtime/model/worker/report/controller为本轮显式实现；无sys.path改动、无旧项目隐式导入。
- 纯源缓存：20261018T125337Z_pure_source_backbone_c1d249f6/artifacts/prepared.pt；源行和valid保持不可变。
- 目标缓存/清单/数据来源审计：20261018T164924Z_multifactor_diagnostic_v1_4fc8f13d，已观察诊断数据。
- 对照logits/共错mask/PCA/历史前沿：20261019T094400Z_progressive_view_v9_9ee04ba2；只读复用，不算本轮重新训练重复。
- 输入源路径解析由configs/datasets.json核对。CNN为新小型通用结构；不声称官方DF/RF复现或语义独立。
