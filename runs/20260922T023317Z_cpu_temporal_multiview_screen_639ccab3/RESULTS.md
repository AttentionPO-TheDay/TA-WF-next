# 实验结果

## STOP：source/valid方向内容重复

冻结结构检查在训练前停止。固定抽样的source与official valid有2条前5000方向SHA-256重复；所有未来日期相对source/valid以及日期间的交叉均为0。没有启动模型、生成checkpoint、预测或性能指标，未来标签仅用于既定分层抽样和类别完整性检查。

停止符合PLAN中的“源/valid/未来选中方向哈希交叉必须为0”条件。下一版本须在抽样前以完整official valid方向哈希排除source候选，并在source内对方向哈希取最小行号作为canonical；不能把这次失败隐去或直接修改冻结清单后续跑。

执行耗时50.84秒，峰值RSS3319912 KiB，GPU隐藏。此结果不涉及生成器效用判断。
