# 论文一实测机理图

依赖纯净时序核心矩阵完成及postrun_audit通过，不启动额外训练。三图固定规则：三日期coadapt/gate支持集logit范数（每10步、0–800、3seed均值±SD）；Day270终端query熵分布与预设0.2/0.4 nat门控区域；全部seed/date的NAR-Recovery散点和方法等权均值。全部生成PDF/SVG/PNG与数值JSON，不预设发散、98%低熵或最低NAR。同步导出五条件主表CSV和Markdown。预算300秒/0.1GiB，无query选模。代码/配置/输入由核心freeze锁定；输出记录自身散列，失败保留。
