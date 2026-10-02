# 20260928T114412Z_source150_separate_learning_rates_944201ea

问题：在固定每类150条、1层局部 Transformer、无残差/锚定/dropout=0.1的条件下，生成器与分类器使用分离学习率能否提高源期验证性能？

数据与权限：Proteus TemporalDrift 的固定 source150（102类×150）用于 source CE 梯度；固定 valid=510 仅用于20个预定选模点和评价。输入为已有已核验的方向序列生成器 batch；不访问 Day14、其他未来日期、WTT-Time 或 AWF。source150 是上一轮已登记样本清单的复用，不重新划分 valid。

条件（四个条件均为本轮新训练）：

| 条件 | 分类器 LR | 生成器 LR |
|---|---:|---:|
| A_1x1x | 1.0×基础 | 1.0×基础 |
| B_1x0.3x | 1.0×基础 | 0.3×基础 |
| C_1x0.1x | 1.0×基础 | 0.1×基础 |
| D_0.3x1x | 0.3×基础 | 1.0×基础 |

所有条件共享每个 seed 的模型初始化和 batch 索引流；基础 schedule 为 step 1–6400/6401–9600/9601–12800 的 0.001/0.0003/0.0001，两个参数组分别乘上条件倍率。每个 seed 为21729、23407、22026，batch=64，12800步，20个固定 valid Macro-F1 选模点（3680至12800，每480步），AdamW、weight_decay=1e-4。生成器仍由 source CE 端到端更新；valid 不反向传播。

主指标：固定 valid accuracy 与 Macro-F1，辅助报告 source accuracy/F1、source-valid差距、生成器 token/参数变化及逐 seed 一致性。候选保留门槛：相对 A 平均 valid Macro-F1 至少+1.0pp、三个 seed 均为正、平均 accuracy 不下降，且 Macro-F1 不出现条件性下降；否则不采用。

资源：CPU，最多6并发、每任务2线程、单任务14400秒、pipeline 30000秒。每个实验目录保存配置、代码哈希、checkpoint、预测、日志、核验与汇总。

限制：分离学习率同时改变生成器/分类器优化轨迹，不能单独归因于某一模块；固定 valid 已参与方法开发；本轮只检验优化速度错配假设，不宣称抗时间漂移。
