# 显式复用来源

- v12 scores_query/valid 缓存提供冻结候选、原生预测、逐行标签、v9_margin 和 margin_plus_soft；cells.json 提供原诊断阈值及计数；source_calibration.json 提供已冻结的15视图权重和温度。
- v10 direction_iat 原始视图 logits 与 v11 四种扰动 logits 提供辅助证据；只读并逐行对齐，不调用模型。
- 本轮 audit.py 为独立缓存分析实现；沿用 v12 的三类改写定义和完整同分阈值协议，无旧训练代码导入或 sys.path 修改。
- 历史来源的实际输入散列列于 artifacts/freeze.json；旧产物通过 v12 的冻结输入及交付清单核验。所有输出归属本 run。
