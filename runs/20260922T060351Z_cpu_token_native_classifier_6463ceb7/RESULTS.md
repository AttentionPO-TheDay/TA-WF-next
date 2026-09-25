# 实验结果

完成：生成器run token作为分类模型唯一输入，训练/推理均不使用原始packet或辅助回归。source2040/valid510，三seed各15epochs，CPU4线程。

| 输入token | valid accuracy均值±SD (%) | macro-F1均值±SD (%) |
|---|---:|---:|
| exact方向+连续log长度 | 6.14±0.63 | 2.61±0.30 |
| coarse方向+离散长度bin | 23.01±0.97 | 20.49±1.49 |

coarse三个seed accuracy22.55/22.35/24.12%，macro-F1 19.35/19.95/22.18%。exact对应6.86/5.88/5.69%、2.96/2.47/2.40%。方向embedding、长度编码和位置embedding后接保序卷积及分段池化，可直接训练得到coarse-token分类信号。这验证了用户要求的直接token训练流程，尚非生成器相对其他输入的优越性证明。

exact和coarse长度编码分别是连续值线性投影和类别embedding，优化/表达方式不同；不能由本实验断言粗化信息优于精确信息。exact模型辨别力弱，应优先诊断编码和训练动态。历史packet约23.79%仅供背景，未作为本轮同架构对照，不能声称等价。valid参与选模，为开发结果而非独立测试；未来日期未读取，未评价漂移模块。

实现范围：只使用生成器方向与长度字段，未使用全部多视角token；512run截断，coarse无精确长度旁路。exact参数79094、coarse79286。原先run CNN实验也已直接输入生成器字段，本轮主要新增embedding、位置编码及分段读出，不宣称首次直接训练。

完整性：两个合成测试覆盖coarse碰撞/精确区分、padding及embedding梯度；6行指标3060预测独立重算、选模、hash、source/valid隔离及checkpoint重载均通过。日志耗时63.10秒，峰值2094444 KiB，全程GPU隐藏。抽样、token、模型、代码和配置hash均保存在本run。
