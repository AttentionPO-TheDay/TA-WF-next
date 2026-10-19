models.py与metrics.py从v14显式复制；inference.py仅提取v14 worker.py的扰动、加载、冻结前向与margin函数，去掉旧角色选择入口。CE拟合/公式沿用v14，数据角色与核验本轮实现；不通过sys.path导入任何旧训练代码。所有checkpoint路径显式配置，零新神经网络训练。
