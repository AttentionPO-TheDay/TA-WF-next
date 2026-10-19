# 论文一纯净时序矩阵图表 v2

问题：已审计的五条件矩阵显示怎样的 logit 尺度动力学、Day270 终端预测熵分布与 NAR–Recovery 权衡？

输入固定为 `20261018T144952Z_pure_temporal_core_matrix_v2_4aa89f43` 的 45 条指标、36 组新适应动力学、Day270 的 15 份终端预测，以及九个历史 frozen 引用。读取前核验核心 `postrun_audit.json` 和 `freeze.json`；逐一核对预测 SHA256。旧图表 run `20261018T144246Z_pure_temporal_mechanism_figures_e95fa886` 绑定另一套含 `tent_coadapt` 的矩阵，其图、表和数值均不复用或覆盖。

图1：三日期分面，四种适应在固定 1024 条无标签 reference 上的 step0–800 平均 logit L2；每 10 步探测，三个训练 seed 等权均值和样本 SD。`frozen` 的 step0 固定水平作为参照。图2：Day270 五条件终端独立 query 熵直方图；每个条件内先按每 seed 归一化再对三个 seed 等权平均，门控区间 0.2–0.4 nat 仅标注，不声称 query 参与适应。图3：每条件九个 seed×日期 NAR–Recovery 散点及等权均值。五条件主表按日期汇总三 seed 的均值与样本 SD，并报告全部九单元的等权均值；不做数据驱动挑选曲线或条件。

输出仅写入本 run：三图各 PDF/SVG/PNG，数值 JSON，主表 CSV/Markdown、散列清单、RESULTS。原始数据、权重和核心 run 均只读。本实验不训练、不重新推断、不选模；query 标签只通过已审计的最终指标间接使用。时间预算 300 秒，输出 0.1 GiB；任一哈希或 45 单元完整性检查失败即停止，不输出部分结果为完成。
