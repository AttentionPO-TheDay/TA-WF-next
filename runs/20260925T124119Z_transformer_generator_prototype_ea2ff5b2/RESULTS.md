# 实验结果

状态：completed；原型结构检查通过。

已实现 [transformer_proto.py](/home/rbf/TA-WF-next/src/ta_wf_next/transformer_proto.py)：

- packet、exact run、window 三类 token 的独立投影和 type embedding；
- CLS + 小型 Transformer encoder；
- 同一原始 packet span 在三种视角中同时遮挡的无标签重建目标，避免从另一视角直接读取答案；
- 少量标签的显式 CE 微调 step；
- 零初始化 bottleneck adapter；
- 冻结 teacher、置信度筛选 KL、双视角表示一致性的 adapter-only TTA step。

`batch_from_views` 直接消费本项目 `traffic_views.py` 的 `TrafficViews`，同时保留每个 token 的原始 packet span。TTA helper 不接受 query 标签，并强制 teacher 与 student 分离、teacher 冻结、student 非 adapter 参数冻结。

## 核验

执行 `PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python -m unittest discover -s tests -v`：43 项运行，其中 42 项通过、1 项既有可选 legacy parity 检查跳过。新增测试覆盖：

- 合成短流量和完整 5000 包预算的 token shape/forward；
- 跨视角 span mask、padding mask 和重建梯度；
- 少标签微调梯度；
- adapter 初始恒等映射；
- teacher confidence selection、teacher/主干冻结和 TTA 梯度。

结构复核发现若按 batch 内最大 run 数 padding，window token 的绝对位置会随同批其他样本变化；已在评分前改为固定观察预算下的各视角固定槽位，并增加“单独输入与混合 batch 输出一致”测试。这是原型实现修正，不存在真实数据分数可供返调。

本轮没有读取真实数据、训练分类模型、选择 epoch、访问未来日期或报告漂移收益。因此这只能证明实现闭环可运行，不能证明 Transformer、蒸馏或 TTA 有效。真实实验仍需另立冻结配置，尤其要规定无标签数据权限、扰动与 span 规则、伪标签阈值、适应步数、回滚条件和日期评价角色。
