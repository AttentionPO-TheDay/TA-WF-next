# 来源与复用边界

base_model.py/view_base.py/encoder_base.py/common.py字节复制前并行方向run同名文件；candidate_base.py复制其model.py。model.py仅将多尺度生成器TAM缩放改为显式raw/log1p选择，不复制RF架构。worker.py据前run worker显式改写，所有候选施加同一既定mask并按两配方构造优化器；无旧项目sys.path或训练代码依赖。

prepared.pt字节复制已审计source/valid缓存。log_current复制前run span_mask三个已完成任务，预测、初始state、索引、曲线标记历史复用；checkpoint仅重载评价审计，不参与新训练初始化。RF源码仅用于静态差异审计，原native_prepared只用于source/valid行/标签/TAM一致性核验，无新raw/future访问。

RF源代码及配方在审计表明确第三方来源，借用Adam/LR/decay的候选不称原创或官方RF完整复现；本轮保留自有生成器与Transformer。Python旧venv只第三方环境依赖。
