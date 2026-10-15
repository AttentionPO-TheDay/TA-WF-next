# TENT BN baseline v2（启动判定修复）

## 问题与范围
原版TENT的BN仿射参数纯熵更新在当前wide+BN Transformer上的表现。TemporalDrift Day14/90/270、seeds 21729/23407/22026；9个800步任务，另有9个仅BN批统计推理控制，无新增基础模型训练。历史frozen/no_pseudo_fixed 18行只读参考，不计独立重复。

## 算法及预算
固定Adam lr=0.00025、betas 0.9/0.999、无weight decay；batch128、support1024、800步、固定随机流。官方Tent forward_and_adapt；源码版本见upstream.json，只扩展官方3处BN2d类型判断以支持BN1d。仅BN affine可训练，LN和所有其他权重冻结，running_mean/var=None，track_running_stats=False。纯目标熵，无源CE、MMD、伪标签、边缘熵、锚定、教师或新机制。末步模型固定，不进行标签选模。

## 数据、标签与评价
共享数据只读，沿用prepared文件与已审计support/query精确split；source/valid仅核验切分，不进入优化。support不包含标签。query不参与任何梯度或优化，不作参考库；query标签只在预测完成后计算指标。显式授权本次BN query batch-stat前向，固定query顺序和batch128；这是批统计前向传导评价，区别于旧基线tracked BN和官方同批测试更新。最终query关闭dropout，BN仍用批统计。bn_batchstats_no_update使用相同query前向规则但无任何更新，用于分离归一化变化和熵更新效果。

## 指标、限制与停止
报告每seed/date与等权9单元均值的Macro-F1、accuracy、NAR、recovery、相对两个参考和推理控制的配对差、预测类别数量/最大类占比、可训练参数、成本。预期不预设失败；失败不支持泛化到全部TTA的结论。单任务上限2400秒，整批3小时，单GPU进程，新增输出<1GiB。结构预检及2步support smoke不读query性能；先冻结再执行。非有限loss/梯度失败时保留产物，不暗中改参。WTT-Time/AWF及非时间因素关闭。

## 修复记录

本版本保留原算法、数据、划分、预算和选择规则，仅修复启动判定：预检允许写入 `artifacts/events.jsonl`；正式 runner 要求 `launch.json` 存在并写入一次性 `runner_started.json`，防止重复启动。原失败版本 `20261014T050956Z_tent_baseline_2805cba7` 保留。
