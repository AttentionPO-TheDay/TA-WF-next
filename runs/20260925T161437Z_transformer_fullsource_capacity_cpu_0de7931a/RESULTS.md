# 实验结果

最终状态（2026-09-26）：按冻结配置并行执行平铺多视角与层次化多视角 Transformer，各计划三 seed、100 epochs；仅使用 TemporalDrift source/valid，不访问未来日期或外部测试。层次化三 seed 完成；平铺条件因每条件 5400 秒上限，在 seed3407 完成第30轮日志后、seed 完成前抛出 TimeoutError，seed2026 未启动。实验按预定预算停止，不能宣称完成六组对照。

层次化条件三 seed 已全部完成。按 valid Macro-F1 选出的 epoch 为 80/80/95（seed 1729/3407/2026）；valid accuracy 为 28.627/26.667/26.667%，Macro-F1 为 25.794/24.887/25.024%。均值 valid accuracy 27.320%、Macro-F1 25.235%；source accuracy 55.850%、Macro-F1 54.458%。参数量 113670，valid 预测类别 96/99/100。与历史 DF-only 的 valid accuracy 49.150%、Macro-F1 48.144% 相差 −21.830/−22.909 个百分点；历史 DF 使用不同架构、GPU 45 epochs 和不同选模机会，只是性能标尺，不是严格同轮对照。

平铺条件仅 seed1729 完成：best epoch95，source accuracy/Macro-F1 为 59.412/58.463%，valid 为 27.451/25.854%，预测类别 100，参数 110400。该单 seed 与历史 DF-only valid 49.150/48.144% 相差 −21.699/−22.290pp；不能与层次化三 seed 均值作稳健结构比较。seed3407 仅有每10轮日志、最高已记录 epoch30 valid accuracy 19.412%、Macro-F1 15.800%，未保存可选 checkpoint；seed2026 未训练。

最终独立核验：已完成的 4 checkpoint × source/valid 共 8 组预测、指标、选模 epoch 与重载复算通过，输入 SHA-256 匹配。完整 `verify.py` 报告 8/12、两个 flat seed 缺失，按其完整性断言退出 1；这是预定时间上限造成的缺失，不能写成 12/12 或零错误完整实验。日志见 `logs/flat_multiview.log` 与 `logs/hierarchical_multiview.log`，检查摘要见 `artifacts/integrity.json`。

结论：就已完成的全 source 标签训练而言，层次化模型 valid Macro-F1 约25.2%，平铺单 seed约25.9%，仍与历史 DF 的48.1%有约22–23pp差距，未达到用户要求的相近网页指纹分类水准。层次化 source→valid accuracy 落差约28.5pp，平铺单 seed约32.0pp，显示当前输入/模型/优化组合的泛化不足，但本轮不能把差距唯一归因于任何一个因素。不要继续加入生成器预训练、蒸馏或漂移模块来解释这个基础分类缺口；如继续，另立有界实验检查保序 packet 表示与计算可行性，不能在本轮结果上临时改预算后宣称同一预注册实验完成。
