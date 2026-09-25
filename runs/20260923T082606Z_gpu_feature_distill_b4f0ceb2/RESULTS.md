# 实验结果

状态：completed。GPU 0，三 seed；一个从头训练的融合教师/seed，各 45 epochs；普通 CE、教师 packet 特征对齐、教师 window 特征对齐三个 packet-only 学生/seed，各 45 epochs。source 2040、valid 510、五日期各 2040，固定采样清单。学生推理仅用 packet，教师和训练期投影均不参与推理。约 43.9 秒。

## 主结果

教师 valid macro-F1 为 32.708/33.814/32.814%，均值 33.112%。学生 macro-F1（%，三 seed 均值）：

| 数据角色 | 普通 CE | 对齐教师 packet 特征 | 对齐教师 window 特征 | window−CE | window−packet 对齐 |
|---|---:|---:|---:|---:|---:|
| valid | 24.873 | 25.046 | 25.296 | +0.423 | +0.250 |
| Day14 | 22.458 | 23.344 | 23.134 | +0.677 | -0.209 |
| Day30 | 19.853 | 21.408 | 20.809 | +0.956 | -0.598 |
| Day90 | 16.776 | 17.498 | 17.231 | +0.455 | -0.267 |
| Day150 | 15.146 | 15.450 | 15.421 | +0.275 | -0.030 |
| Day270 | 13.464 | 14.144 | 13.799 | +0.334 | -0.345 |

window 对齐相对 CE 的 valid 三 seed 差为 -0.184/+1.638/-0.186pp，仅 1/3 正；相对 packet 对齐为 -1.421/+2.585/-0.413pp，也仅 1/3 正。虽然 window 对齐的五日期均值相对 CE 都为正，但与同教师、同维度、同损失的 packet 特征对齐相比，五日期全部为负。因此未通过预定的训练期 window 特异收益门槛。对齐 packet 特征相对 CE 的五日期均值均为正，提示“一般教师表示监督”可能有效；但 valid 仅 +0.173pp，不宜作强结论。

## 解释与限制

训练第 45 轮，window 对齐的特征 MSE 均值约 0.31，packet 对齐约 0.15；window 特征更难由当前学生表示拟合，同时 window 对齐的训练 accuracy 约 38.6–40.9%，不高于 packet 对齐的约 41.2–43.3%。这支持“固定同权 MSE 造成表征转移困难/任务干扰”的诊断线索，但不能从训练损失单独判定因果。不能据此认定所有训练期生成器方案失败，也不能在同一 valid 上事后挑损失权重后称其为独立验证。

本轮教师由 window token 获得较强 valid 分类，但 token 的专属中间表示没有可靠转移到 packet-only 学生。与前轮 logit 蒸馏阴性一致，当前可确认的生成器正向证据仍是“训练及推理均保留 token 分支”的融合模式。若必须训练期独用，后续需要新的、事先限定的转移机制与更强的 packet 学生对照；继续微调此 MSE 权重会加重同一小 valid 上的方法搜索。

TemporalDrift 日期均为方法开发评分，WTT-Time/AWF 未访问。三个训练 seed 不构成独立数据集验证，融合教师本轮从头训练，未加载此前 checkpoint。

## 核验

`artifacts/integrity.json`：63 组学生预测/指标与 checkpoint 重载重算、18 组教师 source 特征 target/统计核验，source/valid 方向交叉 0、错误 0。教师和学生均核对 valid 最早最佳 epoch。输入/代码/配置/标准化散列在 `artifacts/pretraining_seal.json`；预测及教师 source target 散列在 `artifacts/manifest.json`。原始数据只读。
