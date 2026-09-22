# exp_6238dacf9aa142cc

## 问题与边界

在已观察的 TemporalDrift 上，以 DF 和方向单分支 Var-CNN 两个不同卷积 backbone 各一个普通源域监督 seed，比较同一冻结模型的 A 原分类头、B 全局类原型、C 全局多参考 1-NN、D 浅层区域匹配、E 同层全局均值近邻。主问题是 D-C 与 D-E 是否跨 backbone、跨日期稳定为正，而非据本轮替 Host 作研究决策。

TemporalDrift 是开发数据；Day14/30/90/150/270 标签只用于全部无标签推理结束后的评分与诊断，不参与训练、选模、reference、片段选择、阈值或超参数调整。source-holdout 是同时间隔离对照，不是全新确认集。

## 冻结数据与输入

- 数据根：`/mnt/data2/ren/datasets/TemporalDrift`；原始 `train.npz`/`valid.npz` 和五个 `day*.npz`，键 `X,y`，102 类。
- 输入：有符号时间戳只取符号（正为 +1、负为 -1、0 保持原值；局部有效性检查将零视为未知位置），截断到前 5000；不把数值当包大小。
- 冻结前内容审计发现官方 train 的 19,439 行中有 718 个完全重复行、且初始按行候选清单产生 144 个跨角色重复组；该 v1 清单明确保留为无效审计证据，未用于训练。修订不依据任何未来结果。
- v2 对 train 内重复完成去重后，进一步审计发现官方 valid 与 v2 train/reference/holdout 分别共享 152/1/15 个 admitted-input trace，故 v2 也未用于训练。最终 v3 先排除所有与 official valid 内容重叠的 train 行，再对剩余 admitted input（前 5000 个值取 sign 后的 int8 bytes）分组，每组只保留最小 row index；这些重复组没有跨标签。随后在每类 canonical 行内以 NumPy PCG64 permutation 排列：前 2 条 reference、随后 20 条 source-holdout、其余 supervised-train。官方 `valid.npz` 整体作为 source-only validation；其内部 9 个重复副本不跨角色，但作为选模权重限制记录。最终角色按索引与 admitted-input 内容均互斥，reference 每类相同预算。
- 最终清单在 `artifacts/splits_v3.json`，正式内容复核为 `split_content_audit_v4.json`；更早版本均是阻止数据泄漏的失败证据，不删除。NPZ 没有 session/site 字符串 ID，样本 ID 明确定义为源文件名与 row index；这一限制保留。

## 模型、选模与预算

- DF：本仓库 `DF`，输入 `[B,1,5000]`。Var-CNN：本仓库明确标注的 `VarCNNDirection`（原 Var-CNN direction encoder 的单分支派生，不冒充完整双分支；因为跨数据集统一方向输入且原始第二通道是时间值）。
- 两者均 seed 6238、从头初始化、AdamW(lr=1e-3, weight_decay=1e-4)、batch 64、最多 30 epoch；仅按 source-only validation macro-F1 选 best epoch。无未来数据训练或选模。每 backbone 最多一次新训练，总计最多两次。
- 旧 checkpoint 不复用：已定位 DF checkpoint 使用完整官方 train（会包含本轮 reference/holdout）且多为适应实验；未找到协议相符 Var-CNN checkpoint。
- 先用合成与极小真实索引 smoke；完整 GPU 运行只在 smoke、数据与 GPU 检查通过后启动。停止边界为本轮两个 seed，不扩展模型、数据集、few-shot 或超参搜索。

## A/B/C/D/E：评价修订 v4

- A：best checkpoint 原分类头。
- B：L2 归一化全局 embedding 后，每类对同一 2 条 reference 求均值并再归一化，余弦最大类。
- C：同一全局 embedding 对全部 reference 作余弦 1-NN，继承参考标签。
- D：固定浅层卷积图，逐样本要求每个位置的完整理论感受野均处于 0..4999 且覆盖的输入均非零、有限。将合格位置按顺序等分为 4 组，组内均值并 L2 归一化；匹配规则仍为 query 区域对 reference 区域最大余弦的均值，再选最高分 reference。
- E：使用 D 的同一卷积层和同一合格位置，直接平均原始位置特征并归一化，做同 reference 余弦 1-NN。不平均已归一化的区域描述子。
- DF 层 `feature_extraction.0` 输出 [B,32,1249]、RF=22、stride=4，含端点的 RF 索引为 [4i-8,4i+13]；VarCNNDirection 层 `dir_encoder.convs.0` 输出 [B,64,1250]、RF=35、stride=4，索引为 [4i-17,4i+17]。完整长度下分别保留 i=2..1246、i=5..1245，再按实际输入 mask 筛选。
- D/E 至少需要 4 个有效位置；不足时该 query 的 D/E 均回退到 C。无有效描述子的 reference 从 D/E 中同时排除；若任一类失去全部有效 reference，本批所有 query 的 D/E 均回退到 C，避免悄然缩减类别集。保存有效位置数、实际区域推理 mask 和回退数，分别报告全量及区域推理子集的 A/B/C/D/E 指标。
- A/B/C 和 encoder 训练保持原输入与结构。通过一次前向 hook 提取浅层特征，新增 D/E 不修改分类头。BN 训练统计仍可能包含补零，推理使用冻结统计；mask 只保证所选卷积位置的直接感受野无未知输入，不宣称完全消除训练中的长度影响。
- 本轮检验浅层区域证据的保留与匹配，不把重叠区域解释为独立网页资源，也不声称与旧深层定义等价。详见 `artifacts/local_layer_spec_v4.json`。


## 指标与产物

对 source-holdout、Day14/30/90/150/270 报 accuracy、macro precision/recall/F1、逐网站 accuracy，逐日期 D-C 和 future(D-C)-source(D-C)，增加 accuracy/macro-F1 的 D-E 及相对 source 的差值。报告 reference 原始/全局/局部存储字节、特征抽取和 A/B/C/D/E 合计 wall time、比较次数/等价点积数。D 诊断在预测完成后使用真标签，统计正确同站 reference、错误跨站 reference、reference/local-region 被不同真网站命中的广度及频率，识别非区分性稳定模式；不据诊断返调本版本。

状态：frozen；未来开发评分尚未开始。

## 2026-09-15 训练前修订记录

用户授权修正 padding 与同层对照；本次只做 CPU 审计和 smoke，不启动训练或未来评分。v3 split 的 SHA-256 仍为 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`。

source reference 中 195/204 条短于 5000，中位观测长度 1002；旧深层 RF 1755/1786 无法覆盖大量短 trace。因此按输入结构改用浅层，未按准确率选层。source train/reference/holdout/valid 都未发现末个非零位置之前的零或非有限数；零时间戳与 padding 的语义仍不可凭该检查完全识别，采用保守零 mask。新规则下所有 source 角色样本均有 >=4 个有效位置，102 类 reference 完整覆盖。

旧计划保留在 `artifacts/PLAN_before_eval_v4.md`；旧局部定义和 smoke v3 保留为迁移证据，不再作为当前 D 的规范。结果仍为空，训练 seed、预算、选模、reference 和 split 均未变。
