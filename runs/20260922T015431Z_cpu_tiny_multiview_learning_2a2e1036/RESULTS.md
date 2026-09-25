# 实验结果

## CPU tiny learned probe 完成

2026-09-22。四个独立轻量模型使用相同抽样：source 2040、valid 510、JP 2036、subpage 2040。source 标签用于训练，valid macro-F1 选择 epoch；JP/subpage 仅作已暴露开发评分。CPU 4 线程，AdamW，15 epochs，batch 128，未使用 GPU、WTT 或 AWF。

| 视角 | valid accuracy / macro-F1 | JP accuracy / macro-F1 | subpage accuracy / macro-F1 | 最佳 epoch |
|---|---:|---:|---:|---:|
| packet tiny Conv1d | 26.67% / 23.67% | 21.46% / 19.37% | 11.72% / 9.86% | 15 |
| windows tiny MLP | 29.80% / 27.65% | 23.23% / 22.13% | 11.03% / 10.10% | 15 |
| exact-run masked pooling | 5.88% / 2.24% | 4.72% / 1.73% | 2.99% / 0.91% | 15 |
| coarse-run masked pooling | 4.90% / 2.21% | 5.35% / 1.90% | 3.38% / 0.99% | 14 |

## 结论

这轮提供了packet/window表示可被非线性模型利用的开发证据：在同一抽样上，小型可学习 packet/window 模型高于此前的线性 probe（packet valid 11.57% → 26.67%；window 20.20% → 29.80%）。window 在 valid 和 JP 上高于 packet，subpage accuracy略低而macro-F1略高。此前线性实验没有valid选模，本轮有15次选择机会；架构也不同，不能将差值全归因于生成器，更不能证明生成器优于强packet基线。

exact/coarse run accuracy为约3%–6%，虽高于102类均衡随机基准约0.98%，但macro-F1很低。这个结果说明当前读出辨别力较弱，不能判定 run 生成器本身无效。当前mean/max读出丢失全局顺序，且 run width=512 会截断样本；exact/coarse 没有位置编码或序列模型。

因此，生成器现在可以说“接口已验证，packet/window 有轻量学习效用，run token 学习路径尚未验证”。仍没有证据证明抗时间漂移、分级路由或 selector 有效。

## 完整性

参数量：packet 108326、windows 44006、exact 29990、coarse 30246；因此相同epochs不等于相同模型容量或计算量。

独立验证通过：12 行指标、18344 条预测、抽样与前序实验完全一致，封存哈希与 valid 选模核验均通过，错误数为 0。详见 `artifacts/integrity.json`。本实验的模型参数量、截断方式和读出结构不等价，因此数值用于机制筛查，不是最终模型比较。

三个模型选中预算末轮，训练损失仍在下降，尚未证明收敛；一个seed、每类20条source训练样本仍有限。下一步可在独立冻结实验中检验保序run读出，并扩大source训练量或训练预算；不因本轮弱run结果暂停packet/window强基线。当前JP/subpage是网络/行为条件，没有时间日期评分。

执行记录：首次启动因缺少PYTHONPATH在导入阶段停止，未加载数据；随后加入本项目src后原配置运行成功。运行37.26秒，峰值RSS 2586992 KiB；没有加载旧checkpoint，只借用旧虚拟环境依赖。PLAN和config在数据访问前冻结，训练脚本和表示代码哈希在seal.json中。20分钟墙钟预算未触及，但本脚本未实现自动墙钟终止，后续长运行须补齐。
