# 显式复用与最小实现

- models.py 从 v9 model.py/framework_base.py 提取 wide_bn 的 generator、methods.py 的 encode/QVLoRA/attach 推理路径；保留旧 state_dict 字段和算子顺序，不导入旧训练体系。
- models.py 的 ViewNet、Pool、input_features 提取自 v10 model.py 的 direction_iat 路径；参数不更新。
- worker.py 的 TAM 重建、tiny-view 变换依据 v11 perturb.py，源大扰动是本轮配置指定的新组合；使用同一原始时间戳语义，不伪造包大小。
- metrics.py 是 v12 metrics.py 的显式逐字副本，包含完整 frozen-correct 分母 pAUC、同分前沿及独立指标预检。无隐式旧路径导入。
- v13 的 rows_query 缓存提供逐样本候选确认特征，v12 提供原始排序、温度及权重。目标仅逐样本代入已冻结的源训练模型。
- 原网络权重、适配器、prepared cache、代码来源及旧交付清单均在 artifacts/freeze.json 中冻结；旧目录不写入。
