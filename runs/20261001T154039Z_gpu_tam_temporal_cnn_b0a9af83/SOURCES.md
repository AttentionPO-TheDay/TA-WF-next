# 来源

base_model.py/view_base.py/encoder_base.py/common.py/worker.py逐字节复制前一并行run同名文件。model.py/test_model.py仅把cpu_temporal_cnn条件名改为temporal_cnn，生成器与卷积结构不改。supervisor.py为本run3任务队列，无旧项目隐式导入。prepared.pt完全复制已审计缓存，不访问原始数据或未来。

GPU Transformer baseline的预测、初始state、索引、曲线、报告显式复制前一run baseline，源自多尺度实验；旧checkpoint仅重载审计，无训练warm start。已完成CPU CNN seed21729保留在前一run，属已观察开发证据，不混入本run GPU三seed均值。CPU部分checkpoint与错误/中止日志全部保留。新GPU CNN三seed从头训练，不能称从CPU无损续训。

Python旧项目venv仅第三方依赖；原条件名虽含cpu不意味着模型实现依赖CPU，改名不改变结构。无teacher、蒸馏、未来适应。

prepare.py逐字复制前run，仅供复用合成测试的relative_tam helper，不执行main，不生成相对时间缓存或改变本轮输入。
