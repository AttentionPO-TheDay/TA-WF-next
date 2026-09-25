# 实验结果

训练完成：五条件×三seed各45epochs，CPU3线程、8分08.04秒，峰值RSS2227636 KiB（约2.12 GiB）。source2040/valid510不变，GPU隐藏，未来/WTT/AWF未访问。独立核验状态见artifacts/integrity.json及logs/verify.log。

## 45轮预算结果

各预算内valid macro-F1最大选模，平局最早；单位%，均值±样本SD。

| 条件 | Accuracy | Macro-F1 |
|---|---:|---:|
| run_pair | 22.88±0.63 | 21.02±0.23 |
| window_pair | 30.46±0.57 | 29.04±0.43 |
| fusion无适配器 | 29.87±0.79 | 28.61±1.26 |
| fusion_shared | 29.48±1.18 | 28.54±1.46 |
| fusion_specific | 29.54±0.82 | 28.27±0.90 |

specific-fusion F1差-0.704/-0.401/+0.072pp，均值-0.344pp，未通过候选门槛。shared-fusion平均-0.071pp，两seed负。specific-shared平均-0.274pp，虽两seed正但均值负，亦未通过。上一轮15epochs的specific-fusion +0.411pp未在更充分预算下保持；不能将其当作稳定适配器收益。

fusion-window F1差+0.385/+0.658/-2.317pp，均值-0.425pp；specific-window -0.320/+0.257/-2.245pp，均值-0.769pp。融合均未通过“均值正且至少2/3正”的互补候选门槛；window仍是当前整体最强对照。不是所有seed都window更高，也不是证明融合永远无用。

## 预算与过拟合诊断

| 条件 | 15轮best F1 | 30轮best F1 | 45轮best F1 | 第45轮当轮F1 | 第45轮训练accuracy |
|---|---:|---:|---:|---:|---:|
| run_pair | 21.02 | 21.02 | 21.02 | 18.92 | 94.97 |
| window_pair | 26.35 | 28.92 | 29.04 | 27.74 | 93.35 |
| fusion | 25.69 | 28.46 | 28.61 | 25.88 | 97.92 |
| shared | 26.02 | 28.11 | 28.54 | 25.55 | 98.63 |
| specific | 26.10 | 28.06 | 28.27 | 25.76 | 98.51 |

15→30轮有明显开发选模收益，30→45轮累计best提升较小（window+0.121、fusion+0.148、shared+0.428、specific+0.207pp）。best单调不降是增加选模机会的性质，不能直接证明泛化改善。固定末轮F1仍低于所选best，run和fusion类尤其明显，结合训练accuracy接近95–99%支持后段过拟合迹象；不能以增加训练accuracy为理由无限延长。

所选epoch：run12/10/14；window21/23/38；fusion44/20/18；shared32/20/13；specific34/20/13。无模型选第45轮，但2个模型选36–45末段，不宣称严格收敛。第31–35轮与41–45轮平均训练loss：run0.593→0.254，window0.595→0.354，fusion0.332→0.143，shared0.278→0.106，specific0.286→0.107，训练拟合仍改善但验证收益有限。

所有15模型前15轮history与前run逐项完全相同。此前过程更新中“第15轮表现低于上一轮”的表述混淆了当轮值与截至15轮的best，现明确更正：不是复现失败。前15轮重算属于预算扩展锚点，不计作额外独立重复。

## 结论与下一步

不默认加入共享/专属适配器，也不默认run/window融合优于window。冻结生成器+可训练适配器的实现可用性已经验证，但通用source联合训练没有稳定增量；本轮也未测试“冻结已训练骨干，只更新适配器”或漂移后历史样本更新，不能否定那一用途。

建议停止在同一小valid上堆结构或延长到更多epochs。若下一目标是生成器在漂移中演化，应先明确训练期教师/蒸馏还是推理期保留适配器，并冻结历史更新样本与后续评价隔离协议，再设计受控更新实验。也可先扩大获准source训练样本而不是增加模型复杂度；都应另立预算和对照，本轮未启动。

## 范围与版本

全部沿用前run模型/优化器/seed/样本，仅统一45轮，从头初始化，无旧checkpoint或旧训练代码导入。source-only窗口归一化，valid只选模/评分。约3.1%参数差、非等FLOPs、run512token截断和window5000观察覆盖差异等限制继承前轮。valid已反复机制开发，无独立确认/显著性/抗漂移主张，未训练期蒸馏。

模型checkpoint重载全valid预测一致；合成mask/视角权限/初始化/两步梯度通过；仓库35tests通过1跳过。核验脚本独立重建2550条输入、source统计及标签隔离、15最终指标7650预测、45个预算模型单元的selected/final共45900预测和45轮选模；最终状态见integrity.json。配置/代码/数据输入版本见pretraining_seal.json，预测/checkpoint及预算文件见output_seal.json，预算汇总见budget_comparison.json。来源为本项目20260922T114658Z_cpu_run_window_adapters_v2_148f23cd及token-native抽样清单，数据根configs/datasets.json，旧虚拟环境仅提供依赖。
