# 实验结果

状态：completed。GPU 0，三 seed；两同结构教师各 45 epochs，三个同结构 packet-only 学生各 45 epochs。source 2040、valid 510、五日期各 2040，复用固定清单。学生推理只接收方向 packet，不生成 window、不加载教师。运行约 49.9 秒。

## 教师和学生

教师 valid macro-F1 均值：`packet_teacher` 24.893%，`fusion_teacher` 33.131%。融合教师确实更强；此轮从头训练，其三 seed F1 为 34.13/32.37/32.89%，不是直接使用前轮 checkpoint。

学生 macro-F1（%，三 seed 平均）：

| 数据角色 | 普通 CE | packet 教师蒸馏 | fusion 教师蒸馏 | fusion KD − CE | fusion KD − packet KD |
|---|---:|---:|---:|---:|---:|
| valid | 24.666 | 24.067 | 24.208 | -0.458 | +0.141 |
| Day14 | 22.203 | 22.527 | 22.450 | +0.247 | -0.077 |
| Day30 | 19.919 | 21.048 | 19.999 | +0.079 | -1.049 |
| Day90 | 16.767 | 16.846 | 16.819 | +0.053 | -0.027 |
| Day150 | 15.203 | 14.880 | 14.881 | -0.322 | +0.002 |
| Day270 | 13.614 | 14.162 | 14.324 | +0.710 | +0.161 |

fusion KD 相对 CE 的 valid 三 seed 差为 -0.807/-0.851/+0.282pp，未达预定 valid 门槛。五日期虽 4/5 均值为正、平均仅约 +0.153pp，但相对 packet KD 在 Day14/30/90 为负，五日期平均约 -0.198pp、仅 2/5 正，也未达生成 token 特异收益门槛。Day270 的 +0.710pp 对 CE、+0.161pp 对 packet KD 值得记录，但不足以抵消 valid 和其他日期的阴性证据。accuracy 明细在 `artifacts/metrics.csv`。

## 判断与限制

本轮证明了训练期 window token 可以用于构建更强教师，但固定 `T=2、KD 权重=0.5` 的 logit 蒸馏未把该优势稳定传给 packet-only 学生。不能据教师 valid 33.131% 推断学生可达到相同水平，也不能把前轮“推理保留 token 的融合收益”转写为训练期独用收益。本轮否定的是这个具体教师/学生架构、损失和预算组合，不是所有训练期生成器方案。

可能问题：更强教师的信息依赖 window 专属处理层，而学生只见方向序列，固定软标签难以把教师的表示结构传过去；蒸馏温度/权重也未优化。不能在同一小 valid 上事后调这些参数再把所得称为独立确认。若继续，应先明确是否允许推理保留 token；若必须训练期独用，可预先限定新的结构性蒸馏机制与等预算控制，并仍作为 TemporalDrift 开发实验。

TemporalDrift 已反复用于开发，未来日期也属开发评分。这里没有漂移后 adapter 更新，WTT-Time/AWF 未访问。三个 seed 不是独立数据集重复。

## 核验

`artifacts/integrity.json`：63 个学生预测、63 行指标与 checkpoint 重载重算，6 个教师 source logit 重载重算，source/valid 方向交叉 0，错误 0。教师与学生均核对 valid 最早最佳 epoch。输入版本、配置、代码、source window 标准化与固定清单散列见 `artifacts/pretraining_seal.json`；学生预测与教师 source logits 散列见 `artifacts/manifest.json`。原始数据只读。
