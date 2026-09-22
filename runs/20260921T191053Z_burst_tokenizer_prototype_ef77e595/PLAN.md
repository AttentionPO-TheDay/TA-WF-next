# 20260921T191053Z_burst_tokenizer_prototype_ef77e595

问题：实现共享stored-order方向run生成器，验证可逆性边界粗化及三场景接口；不训练评分

状态：frozen prototype-only，2026-09-22。

实现纯Python stored-order方向run模块；不导入旧模型。精确view可逆，coarse-only view单独输出floor(log2 count)与邻居bin差，不能混入精确长度旁路。预算5000；只消费预算内数据；零padding后非零拒绝，非有限拒绝；边界标记表示可能删失，不推断物理资源。时间通道关闭。

验证：合成边界/穷举方向序列；各抽128个固定等距行于Version048/train、Network/train、Behavior/subpage，路径由configs/datasets.json解析。只读X，未读y/URL，不训练评分，不挑样本/阈值/稳定区域。既有时间问题不排序修复；signed timestamp仅取sign。原始数组不能改写。单CPU；输出代码hash、样本行索引、round-trip、字段/边界检查结果。真实检查是接口验证，不是稳定性或泛化证明。测试后停止，不启动模型训练。
