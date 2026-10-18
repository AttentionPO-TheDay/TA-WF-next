# exp_6238dacf9aa142cc

## 问题与边界

在已观察的 TemporalDrift 上，以 DF 和方向单分支 Var-CNN 两个不同卷积 backbone 各一个普通源域监督 seed，比较同一冻结模型的 A 原分类头、B 全局类原型、C 全局多参考 1-NN、D 固定中间层局部简单匹配。主问题是 D-C 是否跨 backbone、跨日期稳定为正，而非据本轮替 Host 作研究决策。

TemporalDrift 是开发数据；Day14/30/90/150/270 标签只用于全部无标签推理结束后的评分与诊断，不参与训练、选模、reference、片段选择、阈值或超参数调整。source-holdout 是同时间隔离对照，不是全新确认集。

## 冻结数据与输入

- 数据根：`/mnt/data2/ren/datasets/TemporalDrift`；原始 `train.npz`/`valid.npz` 和五个 `day*.npz`，键 `X,y`，102 类。
- 输入：有符号时间戳只取符号（正为 +1、负为 -1、0 保持 padding），截断到前 5000；不把数值当包大小。
- 冻结前内容审计发现官方 train 的 19,439 行中有 718 个完全重复行、且初始按行候选清单产生 144 个跨角色重复组；该 v1 清单明确保留为无效审计证据，未用于训练。修订不依据任何未来结果。
- v2 对 train 内重复完成去重后，进一步审计发现官方 valid 与 v2 train/reference/holdout 分别共享 152/1/15 个 admitted-input trace，故 v2 也未用于训练。最终 v3 先排除所有与 official valid 内容重叠的 train 行，再对剩余 admitted input（前 5000 个值取 sign 后的 int8 bytes）分组，每组只保留最小 row index；这些重复组没有跨标签。随后在每类 canonical 行内以 NumPy PCG64 permutation 排列：前 2 条 reference、随后 20 条 source-holdout、其余 supervised-train。官方 `valid.npz` 整体作为 source-only validation；其内部 9 个重复副本不跨角色，但作为选模权重限制记录。最终角色按索引与 admitted-input 内容均互斥，reference 每类相同预算。
- 最终清单在 `artifacts/splits_v3.json`，正式内容复核为 `split_content_audit_v4.json`；更早版本均是阻止数据泄漏的失败证据，不删除。NPZ 没有 session/site 字符串 ID，样本 ID 明确定义为源文件名与 row index；这一限制保留。

## 模型、选模与预算

- DF：本仓库 `DF`，输入 `[B,1,5000]`。Var-CNN：本仓库明确标注的 `VarCNNDirection`（原 Var-CNN direction encoder 的单分支派生，不冒充完整双分支；因为跨数据集统一方向输入且原始第二通道是时间值）。
- 两者均 seed 6238、从头初始化、AdamW(lr=1e-3, weight_decay=1e-4)、batch 64、最多 30 epoch；仅按 source-only validation macro-F1 选 best epoch。无未来数据训练或选模。每 backbone 最多一次新训练，总计最多两次。
- 旧 checkpoint 不复用：已定位 DF checkpoint 使用完整官方 train（会包含本轮 reference/holdout）且多为适应实验；未找到协议相符 Var-CNN checkpoint。
- 先用合成与极小真实索引 smoke；完整 GPU 运行只在 smoke、数据与 GPU 检查通过后启动。停止边界为本轮两个 seed，不扩展模型、数据集、few-shot 或超参搜索。

## A/B/C/D 与固定 D

- A：best checkpoint 原分类头。
- B：L2 归一化全局 embedding 后，每类对同一 2 条 reference 求均值并再归一化，余弦最大类。
- C：同一全局 embedding 对全部 reference 作余弦 1-NN，继承参考标签。
- D：同一冻结 encoder 的最终卷积图；先排除理论感受野触及输入 padding/边界的位置，再把保留位置顺序等分为固定 4 组、组内均值并 L2 归一化。query-reference 分数是每个 query 区域对该 reference 四个区域的最大余弦再取四个 query 区域均值；选择最高分 reference。规则不读 query 标签。
- DF 层 `feature_extraction` 输出 `[B,256,18]`，有效 stride 256、理论 RF 1786、中心 `213+256*i`，保留 i=3..15（13 个）。VarCNNDirection 层 `dir_encoder.convs` 输出 `[B,512,157]`，stride 32、理论 RF 1755、中心 `0.5+32*i`，保留 i=28..128（101 个）。推导与 shape smoke 另存 `artifacts/local_layer_spec.json`。即使完整 RF 不触 padding，相邻区域仍高度重叠，不解释为独立资源。

## 指标与产物

对 source-holdout、Day14/30/90/150/270 报 accuracy、macro precision/recall/F1、逐网站 accuracy，逐日期 D-C 和 future(D-C)-source(D-C)。报告 reference 原始/全局/局部存储字节、特征抽取和 A/B/C/D wall time、比较次数/等价点积数。D 诊断在预测完成后使用真标签，统计正确同站 reference、错误跨站 reference、reference/local-region 被不同真网站命中的广度及频率，识别非区分性稳定模式；不据诊断返调本版本。

状态：frozen；未来开发评分尚未开始。
