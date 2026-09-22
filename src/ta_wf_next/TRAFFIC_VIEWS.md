# 多视角表示接口 v0

定位：系统考察现有信息，不是“全部拼接即有效”。实现traffic_views.py复用burst_tokens.py，无学习参数、模型、条件路由或标签输入。所有字段是已有数据的派生，升维不增加观测信息。

| 信息源 | 已实现视角 | 单位/有效性与限制 |
|---|---|---|
| 方向与顺序 | packet_direction | 原存储顺序；不是按物理时间排序 |
| 连续同方向结构 | exact_runs、coarse_runs | run数量是观测单位数，不是字节数；不等于资源；精确/粗化分开 |
| 局部方向统计 | direction_windows | 默认固定50/250观测位置窗口；positive_fraction与窗口内转向比例；末窗partial；窗口边界的转向不跨窗统计 |
| 有效性/观察边界 | runs.observed_count/budget/end_reason、run边界 | 审计信息不自动导出给模型；可能含采集偏差；边界不是确知真实结束 |
| signed timestamp | timing（显式enable_timing） | 按原位置对齐的同方向前驱间隔、同run跨度与零跨度标记；单位由调用方声明秒；不是跨方向IAT |
| signed size | size | 绝对size序列、run合计；单位继承数据定义，不与run计数混用 |
| 分方向时间窗 | 尚未实现 | 属于后续候选，需固定时间原点、bin宽、溢出及观察预算，不能以标签挑选 |
| 内容字节/TLS/资源 | 不可恢复 | 当前NPZ没有这些字段，不伪造 |
| URL、类别、版本/日期身份 | 不作为生成器输入 | 仅获准的监督/审计/数据角色用途，不当部署特征 |

## 显式模态与导出

`generate_views(values, input_kind='direction'|'signed_timestamp'|'signed_size', budget=5000, window_sizes=(50,250), enable_timing=False)`。

不凭数值大小猜字段。Proteus用signed_timestamp，AWF方向模式，WTT有符号size模式只是预备接口；本轮未打开后两套数据。没有模态用None，不用0占位冒充观测。输入只有一个有符号通道，当前接口不支持同时独立提供time+size。

`views.select('coarse_runs')`只导出该视角；请求缺失/关闭模态报错。精确/packet/windows与coarse联合导出是显式多视角实验，不可再称coarse-only。窗口含观测数/位置，能暴露长度信息，不应作为粗化消融的隐藏旁路。所有输入只读最多budget个值；零padding后非零、非有限值拒绝，预算外不访问。

## 时间语义

实现不生成跨方向IAT或gap（并非把它们截断为0）。每个方向首次出现没有前驱，interval=None；同方向负差也标None并计invalid。同run任何内部时间下降则span=None，不仅检查首末。合法零跨度与缺失是不同值；singleton跨度0，不生成rate。原始时间未校准，因此这些量仅是发布数据的recorded-time派生，不宣称精确物理时间。

同方向间隔可能跨越中间反向观测，不能称混合序列相邻包间隔。时间信息开启须显式选择且记录，默认关闭；None掩码转换/collator留待模型适配阶段，不能当0直接喂模型。不能把异常计数当已验证漂移检测信号。

## 下一阶段边界

窗口50/250是未调优的工程默认值，不是论文最佳设置。当前只有单trace生成、字段检查与单元测试，没有batch collator、embedding、模型融合或蒸馏。应先冻结视角、预算、选模机会及数据角色，再比较各视角和互补性；不直接堆全部字段并宣称创新或泛化。
