# TENT固定源统计量+LN的熵区间更新门控

## 问题与版本

在已完成的 `tent_fixedstats_ln` 上加入唯一极简改动：每个support batch前向后计算逐样本softmax entropy；仅保留 `0.20 <= H <= 0.40` nats 的样本求平均熵并反传，低熵和高熵样本不产生梯度。固定源BN统计量、LN更新作用域、纯熵目标、学习率和800步预算均不变。阈值以此前无标签support熵尺度为依据取整并在正式运行前冻结；不使用query标签、不扫描阈值、不单独声称Day14一定被锁定。

## 数据与权限

沿用 `tent_fixedstats_ln` 已审计prepared文件、checkpoint、TemporalDrift Day14/90/270、3 seeds、support1024/query独立切分；不重新切分。support标签、query输入/标签均不参与更新或阈值选择，query只用于末步评分。source/valid仅做split核验。WTT-Time/AWF和非时间因素关闭。

## 算法

Adam lr0.00025、betas(0.9,0.999)、weight_decay0；800个候选support batch更新机会，batch128。BN的running_mean/var/num_batches_tracked保持源值且BN eval；6 BN affine、4个Transformer LayerNorm的norm1/norm2及final_norm affine开放，共3072参数。每步记录batch平均entropy、保留样本数、跳过样本数、保留比例；若一个batch无中间熵样本则整步不回传。terminal800固定选模。

## 对照与判据

45条历史参考：frozen、no_pseudo_fixed来自MVI；tent_bn、bn_batchstats_no_update来自TENT v3；tent_fixedstats_ln来自前一组实验；全部只读复用。新方法9个seed×date单元。主筛查为Macro-F1>72%且相对tent_fixedstats_ln的9单元平均提升；同时报告相对原版TENT、NAR、纠错率、保留比例、跳过步数和反向暴露量。联合改动无法分离统计保护、LN作用域和门控各自因果贡献；通过也只支持这个冻结配方。

## 审计与预算

GPU1单进程、CPU2线程；9个新任务、每任务800候选更新、总7200步；预检2步不评query。审计检查阈值/config/code hash、split与训练流、逐样本门控统计、仅3072参数更新、BN缓冲区不变、全部54条指标、9个checkpoint重载和有限logits。失败产物保留，不按结果自动改阈值重跑。
