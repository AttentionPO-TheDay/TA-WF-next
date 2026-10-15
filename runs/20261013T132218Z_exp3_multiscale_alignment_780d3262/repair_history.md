# 实现修订记录

2026-10-13 v1→v2：启动后代码复核发现EMA任务附带了非预期动态LR，teacher浮点BN缓冲未EMA且文字写MSE、代码写KL；初版尺度项仅最终分类特征统计，未对应生成器三层；初版窗口公式/日志定义存在不一致。停止v1及队列。v1已运行的Exp-2单元结果、代码、配置、日志、checkpoint保存在artifacts/pre_revision_v1，不作本机制证据，未根据数值调参。修正保持decay=.99/consistency weight=.5/topology weight=.25/window16 stride4 gain4，明确统一定义并重新做梯度/权限/split预检。v2与v1不是独立重复，不覆盖原始产物。

注册辅助器使用lgs物理目录加logs相对symlink，修复直接mkdir logs导致的创建失败；失败发生在训练前。

2026-10-13 文件写入修复：Exp-3首单元800步结束时tmp→json替换出现PermissionError，未保存模型及预测。保留失败attempt的全部artifacts于pre_write_repair；同配置重试，不把失败更新计独立重复。save_json保留原子替换，并在PermissionError时直接写入已完整编码的JSON；诊断确认直接写入可行。Exp-4同步采用相同I/O修复。独立重载审计显式调用训练入口setup，使线程、确定性、TF32设置与训练预测一致；容差未放宽、学习规则未改。解释器路径更新为现有backup虚拟环境，仅复用依赖环境而非旧代码。
