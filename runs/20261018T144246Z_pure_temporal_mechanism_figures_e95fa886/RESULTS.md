# 机理图生成结果

三张图均导出PDF/SVG/PNG，数值见artifacts/figure_data.json。正文五条件主表见核心矩阵artifacts/official_benchmark_table.csv。

图1：仅固定1024条无标签reference的step0–800评估，阴影为三seed样本标准差；不因probe结果改变训练。图2：Day270终端query熵，0.2 nat为预设低熵阈值；不能从低熵本身推断错误，也不预设98%比例。图3：各seed/date单元的NAR与Recovery及等权均值；不预设校准方法达到最低NAR。

- frozen：Day270 H<0.2占比均值 51.645%。
- tent_coadapt：Day270 H<0.2占比均值 66.307%。
- tent_entropy_gate：Day270 H<0.2占比均值 53.547%。

解读边界：支持集probe与query终端指标角色分离。参数与统计量锁定只验证实现范围，不保证logit范数有数学上界、不自动证明物理因果或零误杀。该结果是已观察TemporalDrift上的开发机制诊断。
