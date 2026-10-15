# 外部迁移数据审计

WTT-Time 与 AWF 的结构、角色、输入模态和标签边界审计。该 run 不训练、不适应、不读取 query 标签做评分。

## 结论

WTT-Time 具备 100 类统一标签、15 个历史训练阶段、5 个开发阶段和 7 个 future 阶段；原始向量为 3000 长有符号包大小，必须使用独立方向序列前端，不能复用 TemporalDrift 时间戳语义。AWF 具备 200 类、5000 长有符号方向序列，训练容器与 5 个相对时间 query 容器分离。Version/Network/Behavior 的既有身份审计显示跨角色共享内容，暂不把它们作为未经新角色冻结的独立确认训练集。

下一 run 将分别训练 WTT-Time/AWF 自有类别模型，固定方向前端和五个适应条件：frozen、tent_bn、tent_entropy_gate（tau=0.20/0.40）、Proteus-default、no_pseudo_fixed。
