# 冻结基线 Day14 / Day90 漂移诊断

状态：frozen。2026-10-06用户授权GPU0开始此前提出的Day14/Day90只读实验。

问题：wide+BN CE三seed源期模型在Day14/Day90的accuracy与Macro-F1及相对源期valid下降多少？只作描述性诊断，不设置方法采用门槛。

模型：20261002T053706Z_progressive_capacity_bn_mixup_1ad233da 的wide_bn，seeds 21729/23407/22026。复制已有best权重，按历史20次valid机会的最高accuracy及最早严格提升复核；本次选模机会0。不选最好seed，不重训，不评价hard-mask或集成。

数据：configs/datasets.json定位TemporalDrift。valid使用原固定510行；Day14/Day90按原顺序全量。标签映射0..101，标签仅结构核对、评分和事后诊断。各日期不建立适配池，不执行TTA、梯度、BN重估、增强或目标调参。其他未来日期/WTT-Time/AWF关闭。结构检查、推理评分分别写入exposure.jsonl。结果按已观察开发证据登记，不能作为独立确认。

输入：原始day14.npz/day90.npz，前5000个有符号时间戳先float32；0作为后缀padding，保留存储顺序。TAM为2×1800原始方向计数，abs值提升float64后floor(t×1799/80)，t≥80进入末bin，log1p由模型完成。不得使用tam_day90.npz。重建valid全部510行与原cache逐元素核对；目标全行使用两套计数实现交叉核对、检查每方向计数守恒。重复指纹与训练/valid进行只读核对，保留全部目标行并报告重叠，不能择优过滤。

指标：三seed各自accuracy、固定102类Macro-F1及source-valid减target的百分点；汇总均值和样本标准差，逐类支持度、预测、logits全部留存。valid是经反复选模的510行开发参照，不称独立Day0测试；日期差异不能独立识别时间的因果效应，也不预设Day90必然更差。

预算：GPU0一个进程，batch64，2 CPU线程，单次流水线最多3600秒，无训练/适配步骤。完成后自动独立重算9组指标并重载checkpoint核验预测。输入/权重/配置不一致、非有限或空样本、未知标签、审计失败即停止，保留错误。启动后不持续轮询。

冻结：源模型/历史选模记录/输入缓存/执行代码/config/PLAN和数据配置在提交前冻结。原始授权文件byte hash及完整行清单在目标评分前登记，并完成结构检查。输出路径均在当前run。
