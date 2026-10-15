# TENT固定源BN统计量 + Transformer LN单组增强

## 问题与版本

v1联合干预：纯softmax entropy、固定源BN running_mean/var、放开6个BN1d affine以及4个Transformer LN和final_norm affine，检验平均Macro-F1能否超过72%及原版BN-TENT。仅本配方9个新适应任务；没有BN-only/LN-only消融，不分离两项贡献，也不宣称参数容量原因或通用使能框架已经成立。此前query反馈参与本配方开发，TemporalDrift为已观察开发数据。

## 输入、数据与标签权限

共享原始数据按configs/datasets.json定位，只读；沿用config列明已审计prepared文件及基线checkpoint。5000有符号时间戳及由其重建的2×1800 TAM，wide_bn/tam_only底座不变，不新增基础模型训练。Day14/90/270；3个源训练seed为21729/23407/22026，各seed/date从自己的源checkpoint独立初始化。每日期沿用已审计唯一输入去重与source排除、support1024、独立query；不得重新切分。source/valid输入只用于身份与split核对，无source前向/损失或标签优化；source row采样仅计算匹配流hash。support标签不进入适应。query输入不进入适应/BN统计/参考库，标签仅末步固定预测后的评分。未访问WTT-Time/AWF及非时间因素。

## 实际算法与选模

Adam lr0.00025、betas(0.9,0.999)、weight_decay0；800步，每步support batch128，drop_last、target generator seed2026，source行核对seed2025。纯官方softmax_entropy(logits).mean()，无源CE/MMD/伪标签/边缘熵/锚定。所有6个BN保持track_running_stats=True且强制eval，源running_mean/var与num_batches_tracked逐位冻结；仅BN affine开放。head.0/head.1的norm1/norm2以及final_norm开放affine；准确参数3072（BN1792+LN1280），其他参数冻结。support时其余模块train/dropout维持原版，dropout seed2024+100000+step；query全model.eval，固定batch128、既有query行顺序，不读query统计。terminal800唯一选模点，无标签早停/调参/选择最佳seed。

## 对照与判据

36条历史参考：frozen/no_pseudo_fixed来自MVI，tent_bn/bn_batchstats_no_update来自已完成TENT v3，精确复用各9条预测，不重新适应。主开发筛查：9单元等权平均Macro-F1>0.72且配对平均比tent_bn提升；报告每seed/date以及平均accuracy、Macro-F1、NAR、recovery、预测类别数/最大类占比和配对差。NAR分母为Frozen正确样本，recovery分母为Frozen错误样本。另提供增强与原TENT的直接伤害/纠错率。BN推理规则是主动干预项，原版使用query批统计的协议差异必须报告。成功只能支持联合增强配方在此开发基准有效；失败也完整保留。

## 资源、预检、审计与停止

GPU1单进程共享（计时只描述）、CPU2线程；新任务9（7200更新、921600次support反向trace暴露），每任务<=2400秒、总<=7200秒、新增输出<=1GiB。预检每条件2步support smoke，不评query；验证BN缓冲区不变、仅3072参数开放、BN/LN各有梯度、其他参数不动，完整800步CPU索引流与既有hash一致；smoke产物隔离在preflight/。代码/计划/配置/输入与参考预测冻结hash。正式结束独立复算全部45条预测指标，检查9个checkpoint全部参数及缓冲区，重载每个checkpoint首128条query logits，并生成RESULTS/verdict/contrasts后再登记completed。非有限loss/梯度、split/hash/作用域/缓冲区失败、预算超限即停止保留失败，不按性能自动改参重启。不持续轮询监督。
