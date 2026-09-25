# 实验结果

完成：五条件×三seed、15epochs、source2040/valid510原清单、CPU3线程，3分20.43秒，峰值RSS2230936 KiB（约2.13 GiB）。独立输入/预测/选模/封存审计通过，0错误。无GPU、未来日期、WTT/AWF或在线适应。核验见artifacts/integrity.json。

## 开发结果

单位%，均值±样本SD；全部checkpoint由valid macro-F1选模，平局最早。

| 条件 | Accuracy | Macro-F1 | 参数 |
|---|---:|---:|---:|
| run_pair：两支均读run | 22.88±0.63 | 21.02±0.23 | 158470 |
| window_pair：两支均读window | 28.56±0.60 | 26.35±0.94 | 153766 |
| fusion：run+window，无适配器 | 27.45±1.22 | 25.69±0.94 | 156118 |
| fusion_shared：共享适配器 | 27.58±0.97 | 26.02±0.93 | 157190 |
| fusion_specific：分支专属适配器 | 27.71±0.99 | 26.10±1.04 | 157222 |

run/window的互补性未通过预注册筛查：fusion相对run_pair的F1三个seed均正、均值+4.67pp；但相对window_pair均值-0.66pp、两个seed负，不能只报优于run便宣称互补。专属适配器融合相对window_pair仍为-0.25pp，两个seed负。

适配器有小幅局部正向：specific相对fusion的F1逐seed+0.447/+0.290/+0.495pp，均值+0.411pp；accuracy均值+0.261pp。shared相对fusion均值F1+0.331pp，三个seed亦均正。specific相对shared仅-0.0013/+0.0224/+0.2168pp，均值+0.0793pp：形式上达到原定“至少2/3正且均值正”的候选门槛，但幅度极小，没有统计显著性或稳定优越性证明，不足以认定分支专属机制胜过通用处理层。

结论：可以保留分支适配器为研究候选，但目前整体识别仍以window单视角对照最好，不应默认用run+window替换它。第一步单run适配器阴性和本轮融合适配器局部正向并不矛盾，后者使用了不同输入/模型；不能称跨实验稳健增益。

## 实际接口与对照边界

两个分支均输出512维，经concat1024后Linear1024-102分类。run为512槽方向/桶embedding加位置、两层32通道k5卷积、16段masked mean；window为240→32→512 ReLU MLP。window输入是50/250窗口的方向比例/转向比例，固定位置缺窗置0，不附加exact run字段、时间、资源或真日期。每个固定位置仅用source均值/标准差归一化（std<1e-6改1），valid复用；不存在未来/valid拟合归一化。

run适配器在位置编码前逐token作用；window适配器在第一层MLP的32维整体窗口表示上作用，不是每个窗口独立Transformer。shared是32→16→32残差（1072参数），specific是各32→8→32残差（总1104参数），末层均零初始化。三种fusion初始预测严格相同；公共编码器和分类头按同seed相同初始化。仅生成规则冻结，模型及适配器联合训练，推理仍保留两个分支。

单视角对照不是小型单分支，而是两个独立编码器读同一视角，使参数接近融合模型，全部参数都参与所用预测路径，无闲置参数凑数。全条件最大参数差约3.1%，不是严格等容量或等FLOPs；run CNN计算明显比window MLP大。共享与专属通过不同瓶颈宽度近似匹配参数，所以二者差不能纯归因权重共享策略。run仅前512run，window覆盖前5000观察，信息覆盖也不完全相同，因此不做run表示普遍劣于window的信息论结论。

## 失败修正及验证

初稿20260922T114213Z_cpu_run_window_adapters_faf3c613在首个前向发生512/128维不匹配，未完成参数更新、未valid评分、未生成checkpoint，已登记failed，原代码/配置/日志保留。修订版补齐无适配器与近似容量对照，先通过合成输入形状、mask、视角权限、初始恒等、共享初始化、适配器两步梯度测试再启动；不是根据评分修正。

训练入口进行了source/valid方向hash与历史清单一致性、隔离、非有限值和checkpoint全valid重载预测核验。35项仓库测试通过，1项跳过。额外audit.py独立核对封存、source-only标准化、原始标签、2550条输入的numpy重建、15指标7650预测及逐epoch选模；其最终状态记录于artifacts/integrity.json，结果均值/配对差见aggregate.json。审计仅重算，不训练或改变选模。

来源：本项目traffic_views.py/burst_tokens.py固定生成器，既有20260922T060351Z_cpu_token_native_classifier_6463ceb7清单（数据根configs/datasets.json），训练骨架沿用本项目CPU试验并显式重写双分支。无旧项目代码/checkpoint依赖，仅旧虚拟环境提供依赖。PLAN/config/train/verify/输入和统计封存见pretraining_seal.json，checkpoint/预测封存见output_seal.json；日志logs_train.log。

valid已多轮用于机制开发且本轮选模，三seed仅训练重复，不是独立确认；7/15模型最佳为末epoch，15epochs不证明收敛。未评价漂移、动态演化、训练专用蒸馏。下一步应保留window强基线，并先检查统一更充分训练预算下上述结论是否保持，不继续在同valid不断加结构；尚未启动后续训练。
