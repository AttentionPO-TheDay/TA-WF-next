# 20261003T071914Z_peer_attribution_b4513513

## 归因问题

上一轮 `mask_peer` 相对 `dual_mask_ce` 提升 1.046pp，但双视图 CE 本身低于单视图 CE。该收益可能来自 peer 一致性，也可能来自两视图联合前向和 BatchNorm/优化变化。本轮固定两视图、联合前向、mask流和训练预算，只改变第二视图是否参与 CE。

## 新条件

- `primary_ce_only_joint`：同一模型同时前向两个独立 span-mask 视图；只对第一主视图计算 CE，第二视图只用于联合前向和 BN/Dropout 状态，不加入损失。
- `primary_ce_peer`：完全相同的两视图和联合前向；只对第一主视图计算 CE，第二视图不计算 CE；额外加入两个视图之间的对称 stop-gradient KL。

主要归因比较为 `primary_ce_peer - primary_ce_only_joint`。上一轮 `dual_mask_ce`、`mask_peer` 和历史单视图 `ce_only` 仅作冻结参考。

## 损失和前向

同一模型接受 batch128 的拼接视图；前64为主视图，后64为副视图，单次前向后拆分 logits。两条件使用相同的 source index stream、主/副 mask流和 dropout/BN前向顺序。

`primary_ce_only_joint`: `L=CE(z_a,y)`。
`primary_ce_peer`: `L=CE(z_a,y)+w(t)T²/2*[KL(stopgrad(p_a)||p_b)+KL(stopgrad(p_b)||p_a)]`。
`T=2`；`w=0` 至3200步，3201–6400步线性升到0.25，之后保持0.25。第二视图没有直接标签 CE；没有教师、EMA或额外网络。

## 数据、权限和判定

TemporalDrift source15300/valid510/102类；source标签只用于主视图 CE，valid只用于预定选模/评价。未来日期、WTT-Time、AWF、TTA关闭。

两条件各运行 seed 21729/23407/22026，共6项新训练。相对 `primary_ce_only_joint`，peer候选须平均 accuracy至少+1pp、3/3 seed正且Macro-F1不下降；同时报告相对历史单视图 CE和上一轮 mask-peer的数值，不把参考结果当作新重复。

## 预算

GPU0，最多2路并行，单项7200秒、总43200秒；FP32、确定性、AdamW和12800步沿用上一轮。损失非有限、初始化/索引不一致、第二视图梯度约束或checkpoint重载失败立即停止。固定视图诊断只读，不参与选模。


## 解释边界和只读诊断

保持上一轮 batch128 联合 BN，不声称保留原单视图 batch64 的全部训练方式。主视图 CE 的反向传播可通过联合 BN 依赖第二视图，故“副视图不参与 CE”只指没有副视图直接 CE 标签项，不表示副视图与梯度完全隔离。

固定mask seed91919用于best checkpoint后valid只读诊断（clean/mask accuracy、预测一致率、标准JS散度，T=1）；不新增选模机会。上一轮worker命名为js_divergence的量实际是反向KL组合，本轮标准JS在冻结前纠正；历史diagnostic按原值保留，不与本轮标准JS直接比较，分类损失和accuracy不受该诊断公式影响。最终主指标是原clean valid accuracy/Macro-F1，评估20步点列表见config。

推理checkpoint、prepared及9项历史参考产物均在本轮冻结清单核验。训练不使用历史权重或软目标；只从头构造并对照initial_state检查初始化一致性。
