# 冻结基线 Day30 / Day150 / Day270 漂移诊断

状态：frozen。用户授权补齐此前Day14/Day90之后的时间漂移曲线。

问题：冻结的wide+BN CE三seed在Day30、Day150、Day270的accuracy/Macro-F1及相对valid下降多少？只作开发诊断，不设置方法采用门槛。

模型和权限：复用上一run已验证的source-valid最优checkpoint；三seed 21729/23407/22026；本run不选模、不训练、不TTA、不更新BN。valid标签只作历史checkpoint回放核验；目标标签只作固定102类评分和覆盖核验。

数据：原始TemporalDrift day30.npz/day150.npz/day270.npz全量、存储顺序；使用与上一run相同的前5000有符号时间戳、float32、zero suffix、2x1800 TAM和log1p模型输入。其余日期/WTT-Time/AWF关闭。目标输入与标签均属已观察TemporalDrift开发证据，不能作独立确认。

指标：逐seed及均值/样本SD的accuracy、固定102类Macro-F1、valid减目标百分点；保存全量行号、预测、logits、混淆矩阵和输入hash。目标日期不用于选模或参数调整。

资源：GPU0单进程，最多3600秒；完成后独立重算12组指标（valid+3日期×3 seed）并重载checkpoint核验预测。
