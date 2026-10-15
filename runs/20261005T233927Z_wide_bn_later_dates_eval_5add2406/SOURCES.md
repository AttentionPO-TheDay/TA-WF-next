# 来源

复用上一只读诊断 runs/20261005T233151Z_wide_bn_day14_day90_eval_94f6ef85 的wide+BN CE三seed checkpoint、模型依赖闭包、source/valid prepared输入和审计代码；只扩展目标日期为Day30/Day150/Day270。

本run重新从原始day30/day150/day270.npz重建TAM；不使用tam_day90缓存，不训练、不TTA、不按目标日期选模。
