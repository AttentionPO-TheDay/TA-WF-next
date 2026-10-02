# 来源与复用边界

- `base_model.py`、`view_base.py`逐字节复制上轮同名文件，`encoder_base.py`逐字节复制上轮model.py。来源`runs/20261001T124113Z_gpu_tam_multiscale_generator_101ab9e9`，完整接口继承链见其SOURCES.md。本run model.py仅改变TAM generator中间通道和逐点残差深度，不复制官方DF/RF编码器。
- `worker.py`明确改写上轮worker.py，改变条件构造并增强best时新增generator参数变化审计，其余训练/采样/优化/20次选模/保存重载规则相同。supervisor和audit_historical按上轮队列及审核模式显式修改，不隐式调用旧运行代码。
- `artifacts/prepared.pt`字节复制上轮cache，configs/datasets.json为共享只读来源。原始行/标签/计数及完整valid隔离见manifest.json继承链，无raw/future新读取或重划分。
- 原A d2_w16即上轮multiscale_d139的三个训练seed。audit_historical.py加载旧checkpoint仅复现已有source/valid成绩，复制的初始化/索引/预测/logits/报告标注historical_reuse；所有新条件从头初始化，旧checkpoint不进入训练。
- Python `/home/rbf/TA-WF/.venv/bin/python`仅复用第三方依赖环境（torch2.12.1+cu130），没有旧项目sys.path/训练实现/Tent/donor流程继承。

配置、源码、缓存、历史审计hash见artifacts/freeze.json。独立实现、增益或本因子试验均不能单独认定研究原创性。
