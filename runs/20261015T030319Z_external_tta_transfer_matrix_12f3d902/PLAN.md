# 外部 TTA 迁移性矩阵

## 问题

在不重新调参的情况下，TemporalDrift 上观察到的 TENT 批统计量敏感性、Proteus 过适应和 source-calibrated `no_pseudo_fixed` 行为能否转移到 WTT-Time 与 AWF？

## 冻结输入与模型

WTT-Time 与 AWF 分别训练自有类别模型，不能加载 TemporalDrift 的 102 类 checkpoint。两者共同使用方向序列前端：WTT 的 3000 长有符号包大小只取 sign，AWF 的 5000 长向量取 sign；按位置构造 2×1800 方向直方图 TAM。该前端不使用包大小幅值或伪造时间戳。

每个数据集每个 seed 使用固定 45 个 source epoch、AdamW、0.001 学习率和每类 100 条 source 样本，终止 epoch 45，不用目标标签选 checkpoint。适应固定 1024 条无标签 reference、800 更新、Adam 0.00025。

## 五个冻结条件

1. `frozen`：终端 source checkpoint 直接推断。
2. `tent_bn`：Vanilla entropy TENT，更新 BN affine 并使用 target batch statistics。
3. `tent_entropy_gate`：固定源 BN statistics，更新 BN/LN 归一化 affine，仅保留 0.20–0.40 nats 样本。
4. `proteus_default`：已审计 Proteus port 的 source CE、MMD、熵/边际熵和 GMM 伪标签微调组件，固定低学习率。
5. `no_pseudo_fixed`：source CE + MMD + 熵/边际熵，固定低学习率和 BN 协同更新，不使用伪标签。

## 数据角色

WTT historical dates 只训练，5 个 development dates 用于开发评分，7 个 future dates 作为已暴露诊断；不得根据 future 结果改方法。AWF 的五个相对时间 query 文件使用首批固定无标签 support 与剩余 query，属于 transductive diagnostic，不赋予严格确认资格。Version/Network/Behavior 因既有共享内容审计和标签角色边界暂缓，另立 run。

## 结果与审计

每个数据集、seed、slice、condition 保存 checkpoint、support/query 行号、输入 hash、预测、logits、适应轨迹、BN 状态和运行成本。查询标签只在最终预测后评分。独立审计检查数据角色、无 query adaptation、checkpoint 重载、指标重算和输入前端 hash。
