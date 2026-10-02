# 来源与复用边界

common.py/base_model.py/view_base.py/encoder_base.py/candidate_base.py字节复制前输入配方run同名文件，input_base.py复制其model.py。model.py新实现逐bin通道LayerNorm与两层间3bin聚合；不复制RF骨干，不依赖旧项目代码。worker据前run显式改写，只保留原AdamW/log1p/固定mask，supervisor/preflight/tests适配本轮。

prepared.pt字节复制既有source/valid缓存，路径源自configs/datasets.json；没有新raw数据/重分割/future。flat_none报告/初始state/索引/曲线/预测为前run log_current的明确历史复用，最终origin为原span_mask实验。旧checkpoint只复现已有预测，不作为训练warm start。

旧项目venv仅第三方环境，模块均run-local显式导入；LayerNorm/卷积/聚合是常规组件，本轮独立组合或正向结果不自动证明原创性。
