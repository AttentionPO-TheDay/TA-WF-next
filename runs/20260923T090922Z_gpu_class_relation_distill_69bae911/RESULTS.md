# 类别关系蒸馏：结果

结论：预定的 window 特异候选门槛未通过。教师 window source 类别关系较 packet 分支更有判别力，但将该关系蒸馏给只看 packet 的学生后，valid macro-F1 虽比普通 CE 高 0.654 个百分点（三 seed 均正），仍比同教师 packet 关系对照低 1.265 个百分点；五个开发未来日期相对 packet 关系对照全部更低。相对 CE 的五日期均值仅 Day90 为正。因此不能将本轮解释为生成 token 在训练期独用时已获稳定泛化增益。

## 冻结方案与权限

`PLAN.md` 与 `config.json` 在训练前冻结。复用 TemporalDrift 固定 source 2040、valid 510、Day14/30/90/150/270 各 2040 行；source 每类 20 条。三 seed（1729/3407/2026）× `ce`、`relation_packet`、`relation_window`，各 45 epochs，共同 packet-only 学生、初始权重、batch 顺序和 valid 最早最高 macro-F1 选模。关系条件共享 128→128 训练期投影、scale 10、权重 0.5 的 KL；推理仅需 packet。教师 source 特征来自本项目上一轮已核验的冻结融合教师，未加载旧项目 checkpoint。未来日期只在选模后作已观察 TemporalDrift 开发评分；WTT-Time/AWF 未访问。

## 指标

三 seed 均值；每格为 accuracy / macro-F1，单位 %。逐 seed 原值保存在 `artifacts/metrics.csv`，全部预测在 `artifacts/predictions.npz`。

| 角色 | CE | packet 关系 | window 关系 |
|---|---:|---:|---:|
| source | 42.565 / 40.393 | 48.088 / 46.189 | 41.993 / 39.972 |
| valid | 27.582 / 24.466 | 28.889 / 26.385 | 28.366 / 25.120 |
| Day14 | 25.180 / 22.806 | 26.160 / 23.984 | 24.935 / 22.678 |
| Day30 | 23.121 / 20.977 | 24.346 / 22.623 | 22.778 / 20.876 |
| Day90 | 19.608 / 17.081 | 19.967 / 17.597 | 19.902 / 17.480 |
| Day150 | 17.680 / 15.462 | 17.859 / 15.695 | 17.288 / 15.373 |
| Day270 | 16.389 / 14.369 | 17.026 / 14.825 | 15.588 / 13.749 |

Window 关系减 CE 的 valid macro-F1 配对差依 seed 顺序为 +0.449/+1.245/+0.267pp；减 packet 关系为 −1.701/+0.246/−2.338pp。packet 关系减 CE 的 valid 差为 +2.150/+0.999/+2.605pp。window 关系减 CE 在 Day14/30/90/150/270 的均值为 −0.128/−0.101/+0.399/−0.089/−0.620pp；减 packet 关系为 −1.306/−1.747/−0.116/−0.322/−1.076pp。预定门槛要求 window 同时优于两对照，且未来至少四日期均值为正；实际相对 packet 对照 valid 仅 1/3 seed 正、未来 0/5 日期正。

教师 source 原型关系的真类 top-1：packet 三 seed 为 20.294/20.294/20.294%，window 为 34.265/34.020/35.343%；平均熵分别约 2.503/2.377 nat。第 45 epoch 平均训练 accuracy 为 CE 41.29%、packet 关系 46.32%、window 关系 42.43%；关系 KL 均值分别为 0、0.0573、0.4157。最优 epoch：CE 42/29/43，packet 关系 42/41/43，window 关系 37/45/33。完整历史在 `artifacts/history.json`。全批耗时 35.34 秒，单卡 GPU0；每个学生推理参数 159078，关系条件含训练期投影共 175590 参数。

## 核验与解释边界

独立 `verify.py` 复核输入及上游 feature 散列、固定行号、source/valid 方向重复为零、每类20例及 leave-one-out 原型规则；18 个关系数组和 63 组 checkpoint 预测/指标复算，无错误（`artifacts/integrity.json`）。训练前 seal 未被修改。

较强的 window 教师 source 关系没有转换为较好的 packet-only 学生，可能反映 window 分支所需信息难从 packet 学生表示学习，也可能与目标分布/损失难度有关；本轮不能区分。尤其两关系条件的末轮 KL 规模不同，不宜从固定权重结果断言类别关系蒸馏一般无效。packet 关系条件优于 CE 是这个开发设置的正向结果，但可能主要是同模态教师正则化，不能归功于生成 token。三 seed、小 valid、已观察的 TemporalDrift 日期只支持机制开发判断；不构成外部抗漂移确认。无需据此继续在同一 valid 追调 window KL 权重；若研究训练期独用，应先重审可转移信息与部署必要性，再预定新版本及公平控制。
