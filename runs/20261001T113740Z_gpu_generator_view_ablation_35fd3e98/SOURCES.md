# 本轮显式来源

- base_model.py逐字节复制自本项目`runs/20261001T104959Z_gpu_multiview_generator_head_factorial_cf47d62c/model.py`。这是上一轮独立实现的固定统计+局部CNN生成器与MLP/Transformer模块，未复制官方DF/RF生成器。
- worker.py显式改写上一轮worker.py，只改变视角构造及active/inactive梯度和状态审计；保留初始化、采样、AdamW、学习率、20次accuracy选模、保存/重载规则。supervisor.py据上轮队列与审核模式在本run实现，只调度3条件×3seed并合并3份历史fusion。
- artifacts/prepared.pt为上轮缓存字节拷贝，raw/selection来源与数据结构审计见manifest.json和原run DATA_AUDIT.md。数据行、标签、TAM含义和全valid隔离继承原审计，不新增随机划分或原始数据读取。
- 历史fusion=上轮learned_transformer的seeds21729/23407/22026，已完成训练。audit_historical.py显式加载这三个checkpoint仅复核旧source/valid预测；不会用于新模型初始化。复制的预测、初始state、索引流、logits、报告进入本run并标注historical_reuse。
- Python `/home/rbf/TA-WF/.venv/bin/python`、PyTorch2.12.1+cu130仅复用第三方环境；不设置sys.path导入旧项目代码，不隐式使用旧checkpoint、旧Tent/donor流程。

精确代码/配置/数据与历史审计hash见artifacts/freeze.json。独立实现和本轮增量均不自动构成研究原创性证据。
