# tent_coadapt_gated：全参、目标BN统计与纯熵窄门控

## 问题、版本与数据暴露

唯一新配方v1：全模型参数开放 + tracked BN(momentum0.10) + 目标样本0.20–0.40 nats损失门控 + 纯softmax条件熵。固定Adam lr0.00025，比对已完成BN-TENT与窄门控、no_pseudo_fixed等参考。此前TemporalDrift query结果已参与候选讨论，本实验是开发证据，不是未见确认。阈值已由用户指定，不扫描query。

## 模型与输入

复用本项目已审计wide_bn源checkpoint（21729/23407/22026），模型源码显式复制run本地，不默认加载归档checkpoint。输入5000有符号时间戳及对应2×1800 TAM，102类，与历史参考相同。共享原始数据只读，通过configs/datasets.json定位，prepared来源在config；预处理文件冻结hash。源模型训练已完成，本次仅9个适应任务，不重新训练底座。

## 信息权限与切分

TemporalDrift Day14/90/270，沿用原唯一输入去重、source排除及support1024/query独立索引，不重新切分。source/valid输入仅身份和split核验；train函数只接收timestamps/tam/rows的target support，不接收source样本或任何标签；无source loader、前向或回放。query输入不参与适应、BN统计、门控或选步；query标签只在terminal800固定预测后计算指标。WTT-Time/AWF及非时间因素关闭。

## 实际优化与统计规则

全体model.parameters()进入Adam(lr0.00025,betas0.9/0.999,wd0)，与no_pseudo_fixed优化器参数范围相同；无梯度的inactive branch不强行更新。各seed/date独立从其源checkpoint开始。support batch128、shuffle/drop_last、target generator seed2026，800个候选更新；train dropout seed2024+100000+step。全部模块train，包括6个BN：track_running_stats=True且momentum0.10，不清空running_mean/var；BN每个完整support batch更新一次running统计，query eval使用末步running统计。无source前向。

先完整support batch训练态前向，用逐样本H∈[0.20,0.40]筛选条件熵求均值，仅此loss反传；边界包含。未保留样本则跳过backward和optimizer，但完整前向已更新BN统计。门控仅损失选择：未保留样本仍参与训练态BN批统计及其梯度耦合，不能声称其影响为零或完全防过适应；不新增独立probe/样本路由以保持上一版门控语义。存每步H/mask用于逐样本审计；loss仅entropy，不含source CE/MMD/diversity/伪标签。terminal800唯一选模点，禁止query标签早停和后续阈值调整。

no_pseudo_fixed每步源前向+目标前向、momentum0.10，本配方只有目标前向，所以全参/学习率/BN配置匹配，统计轨迹与标签权限不同；不把这些区别隐藏成“完全相同”。

## 比较与裁定

7个历史条件各9单元共63条：Frozen、no_pseudo_fixed、tent_bn、bn_batchstats_no_update、tent_fixedstats_ln、tent_entropy_gate及tent_relative_entropy_gate；只读复用，不计本次独立重复。9个新seed×date单元。主筛查：9单元等权Macro-F1>72%；72.5%作为额外报告门槛；另报告相对原版tent_bn的F1增益是否>=1.5pp、NAR是否下降（不由F1通过代替安全性）。配对差报告每seed/date，日期和训练seed不算27个独立数据集。NAR分母为Frozen正确，纠错率分母为Frozen错误。即使通过，只能证明联合配方在此开发基准有效，不隔离全参、BN或门控贡献，不证明“支路失步”原因或跨算法普适创新。

## 预检、资源与停止

GPU1共享单进程、CPU2线程；9个新任务×800候选更新，source反向/前向暴露0，target候选暴露921600；每任务<=2400秒、总<=7200秒、输出<=1GiB。预检：历史条件和索引数量/源hash，3日期800步CPU目标索引hash对照no_pseudo_fixed，模型全参manifest、BN状态和momentum、2步support smoke及梯度有限；另全拒绝synthetic batch验证optimizer参数/状态不变（BN统计可变）。预检产物写preflight/隔离。正式结束独立复算72条预测、9个checkpoint重载首128条query logits，核对BN更新次数/梯度作用域、熵mask重算/暴露统计、输入与配置hash；生成RESULTS与verdict再completed。非有限/资源超限/身份不一致即failed保留产物，不自动改参重试。不持续轮询监督，发布后仅核对启动。
