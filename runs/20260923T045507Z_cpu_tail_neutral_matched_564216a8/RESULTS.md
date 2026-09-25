# 实验结果

状态：completed。CPU，3 seeds，15 epochs；两条件完全共享 full source 标准化统计、模型容量和训练预算。

| 条件 | valid | Day14 | Day30 | Day90 | Day150 | Day270 |
|---|---:|---:|---:|---:|---:|---:|
| window_full | 25.363 | 23.822 | 22.836 | 18.845 | 16.398 | 13.934 |
| window_tail_neutral | 26.000 | 24.711 | 22.686 | 19.110 | 16.892 | 14.472 |

tail-neutral 相对 full：valid +0.637pp；Day14 +0.889pp、Day30 -0.150pp、Day90 +0.265pp、Day150 +0.494pp、Day270 +0.538pp。valid 和 4/5 个未来日期为正，达到本轮预设的初步诊断门槛，但幅度较小，不能称为已验证的抗漂移机制。结果支持继续把 tail-neutral 作为候选基线，而不是继续调 adapter。

限制：未来日期仍是 TemporalDrift 开发评分；partial 中和是人为输入变换，可能删除有用的末尾方向信息；本实验未验证在线适配，也未访问 WTT-Time/AWF。
