# 实验结果

多视角生成器v0已完成，保留burst模块为一个分支，不再将burst作为唯一表示。生产实现src/ta_wf_next/traffic_views.py，字段来源与可信边界见TRAFFIC_VIEWS.md。

实现：packet方向、精确/粗化run、固定多尺度方向窗口、显式开启的同方向时间间隔与同run跨度、真实signed-size模式下的size及run合计。缺失模态None；select显式白名单导出，不隐式融合。跨方向IAT/gap不生成；同方向异常时间标None而非截为0。未实现分方向时间bin、batch collator、模型adapter/融合/蒸馏。

10项新增测试通过，覆盖字段数值、跨方向回退、方向内异常、缺失与0区分、size/time隔离、粗化单独导出、预算外不读、padding与非法输入、空trace和零duration、跨模态结构一致性。

全项目32项测试：31通过、1可选旧源码迁移测试跳过。git diff --check与实验注册完整性检查通过。

真实检查Version048/train、Network/train、Behavior/subpage各固定128行，共384行：结构与direction接口一致，0断言失败；同方向间隔异常计数0，未读真实标签/URL。结果见artifacts/real_smoke.json。size分支仅用合成输入验证，不声称WTT实测通过。WTT/AWF未打开。

没有训练/optimizer步骤、分类器评分、checkpoint加载、真实标签选择或稳定性评估。现有模型测试含合成梯度检查，不是训练实验。共享数据未修改。窗口默认50/250未经性能调优；此为接口正确性，而非判别性、抗漂移/泛化证据。
