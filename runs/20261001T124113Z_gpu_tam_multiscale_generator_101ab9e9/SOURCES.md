# 来源与复用边界

- `base_model.py`字节复制上轮视角消融run的base_model.py；`view_base.py`字节复制上轮model.py。来源run为`20261001T113740Z_gpu_generator_view_ablation_35fd3e98`，更早基础实现见其SOURCES.md。本run模型只改TAM局部CNN的跨度和尺度组合，Transformer/统计/读出原样保留，没有官方RF/DF生成器或隐式旧项目import。
- `worker.py`明确改写上轮worker.py，改构造参数为encoder，其余训练/采样/选模/记录/重载规则不变。supervisor/audit_historical遵循上轮队列和审计实现，本轮三条件中的local是历史复用，两个候选各三seed新训练。
- `artifacts/prepared.pt`为上轮同名缓存字节拷贝，shared read-only原始路径来源configs/datasets.json，原数据/行/标签/TAM与隔离审计见manifest.json继承链，不新增raw/未来数据访问或随机切分。
- 原local_d1来源上轮tam_only的21729/23407/22026。audit_historical.py明确读取三个旧checkpoint，仅核验已有source/valid成绩。复制的初始state、采样、预测、logits和报告存入本run，report标注historical_reuse；新训练始终从头初始化。
- Python `/home/rbf/TA-WF/.venv/bin/python`仅第三方依赖环境复用，torch2.12.1+cu130；无旧sys.path训练依赖、Tent/donor限制继承或旧checkpoint默认加载。

源码、配置、缓存及历史审计精确hash在artifacts/freeze.json。独立实现与参数匹配候选实验不能单独证明研究原创性。
