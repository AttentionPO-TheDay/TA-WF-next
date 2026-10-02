# 来源与复用边界

base_model.py/view_base.py/encoder_base.py字节复制上一深度宽度run同名文件。encoder_base.py是前一多尺度生成器的显式本地副本，其内导入均为本run文件，无旧项目运行时代码依赖。model.py在此基线上实现单因素候选；worker.py基于上轮worker显式改写为CPU/GPU统一入口。

prepared.pt字节复制上轮source/valid缓存，路径来源遵循configs/datasets.json；行、标签、原TAM与时间戳不变。relative.pt是本run从同一时间戳按逐trace最大值重建的相对时间TAM，不新增数据或标签权限。

历史d2_w16三seed明确复用，只复算指标与重载best，无warm start。所有24个新任务从头初始化，包括CPU数值对照。旧环境python仅提供第三方依赖；无隐式sys.path到旧项目。

参数增加、初始化、输入与计算变化逐项报告；实现独立与指标增益均不自动证明研究原创性。
