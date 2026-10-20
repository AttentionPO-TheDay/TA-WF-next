# 来源

maintenance_base.py 从已审计v19同名文件显式复制，源头为本项目v17；含固定特征线性头CE+L2拟合与指标/诊断前沿。无旧TA-WF运行时导入。新runner/methods/audit实现五袋保类抽样与路由。本实验只读取config列明的本项目v17/v18/v19冻结产物；共享原始数据由configs/datasets.json登记，本轮不重新读取数组。三个模型seed代表不同源模型，支持抽样不是独立query。
