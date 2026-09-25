# 实验结果

## 保序 run 编码器 CPU probe 完成

2026-09-22。沿用前两轮 CPU probe 的抽样：source 2040、valid 510、JP 2036、subpage 2040。source 标签用于训练，valid macro-F1 选 epoch；JP/subpage 只作已暴露开发评分。前 5000 观测、run width 512、exact/coarse 字段、mask 和 source 标准化均冻结。两个模型分别使用两层保序 Conv1d、mask-aware mean/max pooling 和分类头，15 epochs，AdamW，CPU 4 threads。

| 视角 | valid accuracy / macro-F1 | JP accuracy / macro-F1 | subpage accuracy / macro-F1 | 最佳 epoch |
|---|---:|---:|---:|---:|
| exact-run | 12.55% / 7.57% | 10.27% / 6.36% | 6.13% / 3.67% | 15 |
| coarse-run | 10.78% / 7.40% | 8.45% / 5.27% | 4.80% / 2.92% | 13 |

## 结论

这是 run 生成器的初步正向证据。与上一轮相同输入、相同抽样的 masked mean/max 读出相比：

- exact-run valid accuracy 从 5.88% 提升到 12.55%，JP 从 4.72% 提升到 10.27%；
- coarse-run valid accuracy 从 4.90% 提升到 10.78%，JP 从 5.35% 提升到 8.45%。

两种 run 视角在各角色均高于102类均衡随机水平约0.98%，卷积读出相对历史池化读出均提升。这支持run表示可被学习利用，但尚不能确定提升来自顺序：exact参数从29990增至51558，coarse从30246增至52838，容量和局部组合能力同时改变。需要同容量kernel=1或打乱顺序对照才能归因。exact输入的优势也未隔离通道/容量差异，不能单独归因于精确长度。

subpage 仍较低，说明当前结果不是稳定的跨条件泛化或抗漂移证据。模型参数量、读出结构和 packet/window 模型不同，不能把这些数值解释成最终模型优胜；只有一个 seed、每类20条 source 训练样本，且 exact 最佳 epoch 仍为最后一轮，训练可能未收敛。

## 完整性

独立验证通过：6 行指标、9172 条预测、抽样与前序实验逐行一致，封存哈希和 valid 选模核验通过，错误数为 0。运行墙钟54.74秒，峰值RSS2585948 KiB（约2525 MiB），GPU隐藏。训练脚本由本项目上一CPU run显式复制修改，运行调用本项目生成器，不加载旧checkpoint。历史池化结果为既有对照而非新重复。code/config/model/statistics版本哈希见seal.json。JP/subpage是网络/行为条件，本轮未评价时间日期；run路径值得继续做容量和seed验证，但还不能证明时间漂移收益或分级释放机制。
