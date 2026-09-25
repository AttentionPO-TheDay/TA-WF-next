# 20260922T112458Z_cpu_token_adapter_v1_4a18e272

问题：冻结生成器规则后绑定轻量token适配器是否能稳定提高直接token分类性能

状态：frozen，训练前冻结。用户授权固定生成器+轻量适配器第一步CPU试验。

冻结的是无参数生成规则，不是随机初始化主模型；主模型与适配器在source联合训练。此轮仅run coarse token，未来可扩展window但不在本轮；推理保留适配器，不是训练专用蒸馏或在线漂移更新。不要把联合训练实验称为仅训练适配器。

source2040/valid510沿用token-native run原隔离清单及封存缓存，不重新划分。原数据根由configs/datasets.json定位；源标签训练，valid仅逐轮macro-F1选模/开发评价，平局最早。未来日期/WTT/AWF关闭，无适应。输入前5000观察/512run、方向与分桶，不使用精确长度（缓存仅供完整性核验）。

baseline为上一轮bucket+专属MLP：方向3-16/桶14-16 embedding、桶残差MLP16-16、concat32、位置embedding、两层32通道k5CNN、16段masked mean、Linear512-102。adapter_zero在位置编码之前增加残差适配器z+Linear8-32(GELU(Linear32-8(z)))，上投影权重和偏置全零，使初始输出与baseline一致；adapter_random为完全同参数、同结构的普通随机初始化残差MLP，对照零初始化策略。两个适配器均552参数，总量增幅<1%。不是严格等容量baseline；相对random仅检验初始化，不能证明某种独特适配器结构。公共模块配对初始化、同seed同batch顺序。

三条件×seeds1729/3407/2026，15epochs、batch128、AdamW lr0.001 wd0.0001，CPU3线程，GPU隐藏，timeout600秒，全部从头初始化不加载checkpoint。非有限/隔离/hash错或超时停止。全结果保留，不事后调宽度/epoch。

主比较adapter_zero-baseline，三seed F1均提升且平均accuracy提升才通过筛查；adapter_zero-random同门槛及random-baseline完整报告。保存train动态、参数/耗时、checkpoint、预测、hash。零初始化下首步下投影梯度为零是预期，检查第二步梯度非零。验证初始等价、padding、coarse-only输入权限、全输入重建/隔离/封存、选模、重载及独立指标重算。

实现来源本项目20260922T111009Z_cpu_exact_increment_control_d4ccdda7训练与验证脚本，显式复制并修改模型；旧虚拟环境仅提供依赖，不导入旧训练代码。valid已反复开发、每类5样本、三seed，不作独立确认/显著性/漂移结论。适配器未来演化需要另立数据与更新权限。
