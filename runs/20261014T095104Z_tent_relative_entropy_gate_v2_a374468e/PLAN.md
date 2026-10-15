# TENT相对熵门控（102类归一化阈值）

## 问题与版本

在已完成的固定源BN统计量+Transformer LN熵门控上，只将绝对阈值换成相对熵：H_max=ln(102)=4.624972813284，保留每个support batch中满足 0.1 <= H/H_max <= 0.5 的样本，低熵和高熵样本不产生梯度。对应冻结数值 tau_low=0.462497281328、tau_high=2.312486406642 nats。BN统计保护、LN更新作用域、纯熵目标、学习率、数据划分和800步预算保持一致。

阈值在正式运行前按类别数规则冻结，不扫描query，不使用标签选择；top-1概率与熵不是一一对应，结果只按真实指标评价。

## 数据、权限与对照

沿用上一版已审计 TemporalDrift Day14/90/270、3 seeds、support1024/query独立切分、source checkpoint和输入处理。support标签与query均不参与适应；query仅末步评分。历史参考包括frozen/no_pseudo_fixed、tent_bn、bn_batchstats_no_update、tent_fixedstats_ln及绝对阈值tent_entropy_gate，全部只读复用，不重新适应。WTT-Time/AWF关闭。

## 算法与判据

Adam lr0.00025，800个batch更新机会，batch128；6 BN affine及指定LayerNorm affine共3072参数，BN源running统计量冻结。每步逐样本熵门控，空保留集则跳过optimizer更新；记录保留比例、跳过步数和反向暴露量。主筛查为9单元Macro-F1>72%且相对绝对阈值门控平均提升；报告所有日期/seed、NAR、纠错率、类别塌缩和配对差。该实验只能评价阈值校准，不分离BN保护、LN作用域和门控本身的因果贡献。

## 审计与资源

GPU1单进程、CPU2线程，9个新任务，总7200个候选batch；预检2步，正式后独立审计54条预测（45条历史+9条新结果）、9个checkpoint和重载logits。失败产物保留，不因结果自动改阈值重跑。

## v2修复记录

v1仅在正式runner装载历史参考时失败：6个历史条件各9单元共54行，登记值误写45；无新适应或query评分。v2仅将历史行预算与postrun期望改为54，阈值、数据、算法和选择规则不变。失败目录 `20261014T093930Z_tent_relative_entropy_gate_5d90d230` 保留。
