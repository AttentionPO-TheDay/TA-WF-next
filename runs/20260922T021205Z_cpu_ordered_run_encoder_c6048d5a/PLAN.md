# 20260922T021205Z_cpu_ordered_run_encoder_c6048d5a

问题：保留run token顺序的轻量卷积读出能否恢复exact-run与coarse-run的可学习信号

状态：frozen，数据访问前冻结。CPU 4线程，墙钟上限600秒，以timeout强制。训练seed1729，两模型各15epochs，batch128，AdamW lr0.001 weight_decay0.0001。source valid macro-F1逐轮选最佳，平局取较早epoch。

沿用CPU tiny run的抽样、输入、逐通道source有效token标准化与mask，前5000观测、512run槽。source2040用于监督，valid510选模，JP2036/subpage2040仅开发评分。抽样清单逐行核对前序token diagnostic；不读取其他条件。query标签不用于梯度或选模。预测封存后评分，保存模型、统计、历史和哈希，拒绝覆盖。

实现显式复制并修改本项目20260922T015431Z_cpu_tiny_multiview_learning_2a2e1036/train.py、verify.py，未引用旧项目代码或checkpoint；仅借用旧venv依赖。exact4通道与coarse8通道定义完全沿用前序，coarse不含精确包数；截断末token的next-bin仍可读取观察预算内第513run，此为已声明输入规则。

读出改为Conv1d(C,64,k5,p2)-ReLU-mask-Conv1d(64,64,k5,p2)-ReLU-mask，masked mean/max后Linear128-128-ReLU-Linear128-102。输入padding先清零，每层清零，防止padding污染。没有绝对位置编码，保留局部顺序，不能恢复全局顺序。其他超参不变，重头训练。历史mean/max结果仅作为既有对照，不冒称新重复；参数量增加，因此不能把提升单独归因于顺序。

指标accuracy/macro-F1按角色完整汇报；非有限值或样本不匹配立即失败。一个seed、每类20训练样本，仅筛查可学习性；JP/subpage是网络/行为条件，不是时间漂移日期。不设基于目标结果续训或修改的分支。精确代码版本见seal.json。下一轮扩大数据/预算须另立冻结实验。
