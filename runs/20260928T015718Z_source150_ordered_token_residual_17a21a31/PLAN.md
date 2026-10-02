# 生成器有序方向残差验证

用户授权继续实验验证。当前目标只提升源期基础识别准确率，Day14及所有其他未来/外部数据关闭。上一轮锚定/dropout无收益，本轮保持原dropout0.1、无锚定。

## 问题与三条件

A_base显式复用20260927T032819Z_source_size_local_depth_factorial_dae714b7的B_150_l1三seed（150条/类+每视角1层），不重复训练。B_counts在当前可学习generator token路径增加局部统计残差；C_ordered增加有序原始方向残差。B/C各三seed21729/23407/22026，从头训练，共6个新任务。

当前generator已有两层局部卷积，并保留5个10包子段的次序，不可说它完全没有顺序信息。本轮只检验绕过局部汇聚的直接线性路径能否带来增量。对每50包patch构建100维输入：50个方向值与50个observed mask。C直接保留方向；B把每个10位置子段的方向均值（含零padding、分母固定10）重复10次，原observed mask不变。这样B保留五子段顺序、正负方向计数和有效长度信息，去除该新分支中子段内部的逐包顺序；原生成器卷积分支不改。

两分支均使用共享跨patch的Linear(100,52,bias=False)，5200参数，权重全零初始化。输出加到原generator token，空patch归零；仍100个packet token、52维，run/window与分类器不变。新分支属于generator，由source CE端到端更新。创建分支使用fork_rng保持共同参数与全局训练随机流；B/C全部初始化相同，A共同权重相同，初始输出与A一致。零初始化仅保证起点一致，不保证训练后无损。

参数量A119578、B/C124778；B/C张量形状、参数和更新预算匹配，但B的输入有效秩较低，故C−B是顺序保留/有效输入维度的联合证据，不能唯一因果定位信息损失。历史20260926T024929Z固定保序替换实验阴性保留；本轮是当前可学习路径的增量残差，不是重复其配置，也不默认预期正向。

## 数据权限与配对

复用已审计source150=15300、共同source80=8160、valid510及原行号，prepared/manifest按hash冻结；原始定位遵循configs/datasets.json，不新增原始文件访问、不重新划分、不增加标签。仅source标签进入梯度，valid只选模和评价；source20不用于本轮训练。方向按既有float32后sign/前5000包输入，不引入时间戳或包大小。

同seed共同参数初始化与历史A一致，B/C同seed初始参数和连续排列batch流一致。相同source150、batch64、12800更新=819200次样本呈现，约53.54遍。AdamW lr1..6400=0.001、6401..9600=0.0003、9601..12800=0.0001，weight_decay0.0001。无增强、预训练、TTA、蒸馏或额外损失。

固定3680..12800每480步共20次评价，valid Macro-F1最大、并列最早选best；不依据新反馈追调。报告best/latest source、共同source80、valid accuracy/Macro-F1、CE曲线、generator与新增残差参数变化和固定诊断样本的残差token RMS。诊断仅source。

## 预定比较

B−A检验新增统计路径，C−A检验候选实用增益，C−B检验有序细节相对统计控制的额外收益。候选PASS：平均valid accuracy至少+1pp、三个seed accuracy差均正、平均Macro-F1不降。只有C−A和C−B均过门槛才称支持该保序残差候选；仅胜B而不胜A，不采用；只提高训练分数不算成功。全部报告，无最高seed选择。只有一次数据抽样和已观察valid510，非独立确认/显著性证明。

## 资源与实现验证

6新任务同时CPU×2线程，总12线程、RSS每任务4GiB/总24GiB；每训练job14400秒+180秒收尾，核验900秒，pipeline16200秒上限，预计约2小时，依负载变化。错误/超时保留产物，不自动重启或加预算。启动后不持续轮询，等用户查询。

显式改编本项目已核验run-local脚本，仅导入本项目src，不修改共享代码；显式使用/home/rbf/TA-WF/.venv环境，不加载旧项目模型。训练前完全合成输入检查：B/C参数与初始输出一致、A共同初始化/输出精确一致、新分支梯度非零、旧generator梯度保留、padding不泄漏、counts对10位置内置换不变而ordered改变、子段方向总和/mask保留、batch独立、保存重载。冻结代码/配置/计划/数据/旧A证据后启动。

A复用已通过的best/latest checkpoint重载证据，核对其产物hash与18组保存预测的独立指标计算，不再重复耗时推理，也不称新重载重复。新6任务完成后独立进程重载best/latest×source/common_source/valid共36组预测，核验步数/LR/选模/样本流、全部参数更新、新分支非零、诊断token值。报告36组新重载与18组历史指标复核分列，不混称54个新结果。完成后自动更新RESULTS/STATUS/EXPERIMENTS。

启动检查修正：第一次纯合成检查在第二条件复用基线时遗漏eval()，导致训练态dropout与评价态输出比较失败。补齐每条件base.eval后重新执行；模型/训练算法未变、配置当时仍draft，无真实训练或新性能评价，初次日志保留logs/preflight_attempt1.log。
