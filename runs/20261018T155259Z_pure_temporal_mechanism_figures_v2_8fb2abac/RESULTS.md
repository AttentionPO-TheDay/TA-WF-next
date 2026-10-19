# 五条件机理图结果

基于已审计的45单元纯净时序矩阵；三图各导出 PDF/SVG/PNG，五条件主表在 `artifacts/benchmark_table.md` 和 CSV。输出及输入哈希均通过 render_audit。

图1使用固定无标签 reference 的 0–800 步 probe，阴影为三 seed 样本标准差；图2使用 Day270 终端独立 query logits，三个 seed 等权；图3使用全部九个 seed×日期单元的 NAR 与 Recovery。

- frozen：Day270 终端 query 熵 <0.2 nat 比例 51.645%。
- tent_bn：Day270 终端 query 熵 <0.2 nat 比例 52.661%。
- tent_entropy_gate：Day270 终端 query 熵 <0.2 nat 比例 55.305%。
- proteus_default：Day270 终端 query 熵 <0.2 nat 比例 63.376%。
- no_pseudo_fixed：Day270 终端 query 熵 <0.2 nat 比例 61.973%。

这些图只说明已观察 TemporalDrift 开发数据上的关联与诊断，不证明物理因果。TENT BN 的 query 前向使用批统计量，应单独解释；旧含 tent_coadapt 的图表属于另一套矩阵。
