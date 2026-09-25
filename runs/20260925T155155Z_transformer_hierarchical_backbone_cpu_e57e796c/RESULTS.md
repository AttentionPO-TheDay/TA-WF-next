# 实验结果

状态：completed。CipherSight 启发的“视角内局部编码→视角摘要→跨视角全局编码”基础分类实验，CPU 三 seed×100 epochs 全部完成。6/6 train/valid 预测与 checkpoint 重载复算，0 errors。没有预训练、resource 语义、蒸馏、adapter 或 TTA；没有访问未来日期、WTT/AWF。

## 结论

本轮层次化结构没有改善 valid 泛化，预定候选门槛失败。相对已完成的平铺多视角 scratch 对照，层次化模型的最佳 checkpoint 平均 train accuracy 从 40.07% 提高到 53.20%，但 valid accuracy 从 15.69% 降到 14.38%，valid Macro-F1 从 12.91% 降到 12.50%（−0.41pp）；仅 seed1729 为正，其余两个 seed 为负。更强训练拟合没有转化为更好的网页分类泛化。

| seed | 层次化最佳 epoch | 层次化 train accuracy | 层次化 valid accuracy | 层次化 valid Macro-F1 | 平铺 valid Macro-F1 | 差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 90 | 42.94% | 14.31% | 11.68% | 10.98% | +0.70pp |
| 3407 | 90 | 59.41% | 14.71% | 13.10% | 14.62% | −1.52pp |
| 2026 | 95 | 57.25% | 14.12% | 12.72% | 13.14% | −0.42pp |
| 均值 | 91.7 | 53.20% | 14.38% | 12.50% | 12.91% | −0.41pp |

旧平铺结果来自 `20260925T135213Z_transformer_cpu_convergence_07cd94a5`，是明确复用的已观察开发结果，不是本轮重训的独立对照。两者复用同一固定 prepared 输入、5-shot 清单、三个 seed、100 epoch、每5轮选模、batch64、AdamW 配置；但层次化模型113670参数，平铺110400参数（约+3%），并且局部独立参数/位置编码与注意力拓扑同时改变，不能声称严格等容量或单因素机制归因。

本轮层次化 valid F1 12.50% 还低于既往同5-shot轻量CNN的20.298%，未达到“合格主模型”水平。历史 DF 48.144% 使用全部 source2040 标签，而本轮仅510条标签，不能直接作为同预算达标线；RF 也没有在本工作区的同一固定清单/标签预算下完成可比重训。

## 解释与后续

结果说明单纯将现有摘要 token 分层并不能解决低准确率；更大的 train-valid 差距提示少样本泛化问题，但不能判定唯一根因。当前 packet 每50包压成两个统计量、run 最多128个的输入瓶颈仍未改变；本轮也没有论文的 TLS record、flow 或 resource 语义标注，因此不构成 CipherSight 复现或对其方法的否定。

下一步若继续提升主模型，应先建立全 source 标签、相同输入权限下的 DF/VarCNN 与 Transformer 正式对照，再在固定预算下单独测试保序局部 packet 编码或 run 截断解除。不要以本轮阴性结果为由在相同 valid 上无界叠加结构，也不要启动 TTA/漂移模块。

## 核验与版本

结构测试2项及真实 batch 单步反向传播通过。`artifacts/integrity.json`：6/6预测重算，0 errors，固定输入 SHA-256 `4dc489bf088246c59b73cdb071bb22459c975051ed84b27962bbbf0067d70b24`。层次化主干 SHA-256 `69aad001d0aa82c3d36a73b696b43229b0d52d7487e8e3bfa2aeeae68532bab6`；训练入口 SHA-256 `4f0671fa55e28fa866096b6118b5fd26e1429cdc76bbd908576529279e6f0caf`；配置 SHA-256 `bea70674756bd8c4c9146615ae1c4af8bad9ea82a647511efc35a97c0fee3ea4`。
