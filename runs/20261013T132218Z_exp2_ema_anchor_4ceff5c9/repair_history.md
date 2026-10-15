# 实现修订记录

2026-10-13 v1→v2：启动后代码复核发现EMA任务附带了非预期动态LR，teacher浮点BN缓冲未EMA且文字写MSE、代码写KL；初版尺度项仅最终分类特征统计，未对应生成器三层；初版窗口公式/日志定义存在不一致。停止v1及队列。v1已运行的Exp-2单元结果、代码、配置、日志、checkpoint保存在artifacts/pre_revision_v1，不作本机制证据，未根据数值调参。修正保持decay=.99/consistency weight=.5/topology weight=.25/window16 stride4 gain4，明确统一定义并重新做梯度/权限/split预检。v2与v1不是独立重复，不覆盖原始产物。

注册辅助器使用lgs物理目录加logs相对symlink，修复直接mkdir logs导致的创建失败；失败发生在训练前。

2026-10-14 audit repair：9个任务的训练checkpoint均已存在；重新从每个checkpoint加载student并在固定query split上生成canonical logits/predictions，更新对应指标后复跑独立审计。未重新训练、未修改适应规则、未使用query反馈调参。修复原因是live GPU推理产物与checkpoint重载推理存在约1e-2 logits数值差异；canonical产物以checkpoint重载结果为准。
