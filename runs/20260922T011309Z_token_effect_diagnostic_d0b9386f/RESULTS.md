# 实验结果

## 首轮CPU效用试验完成

2026-09-22。GPU检查三卡98–100%利用率，未启动GPU工作。初始创建说明为仅统计诊断，正式运行前改为冻结CPU prototype读出，变化已写PLAN。没有神经网络训练，但有source监督的标准化与类原型拟合，不能说“完全没有拟合”。

源Network train每类20条（2040），source valid每类5条（510），JP最多20条/类（2036），subpage20条/类（2040）。一个seed1729，输入5000观测。目标标签仅用于预注册分层抽样和最终评分，无目标调参。JP/subpage从此登记为性能开发已暴露；其他国家/Version/Temporal/WTT/AWF未加载或评分。目标分层抽样是离线诊断设计，不是无标签部署采样策略。

## Accuracy（%）

| 系统 | source valid | JP | subpage |
|---|---:|---:|---:|
| Packet direction | 11.57 | 10.36 | 6.52 |
| Exact run（signed log1p count） | 4.12 | 3.93 | 2.84 |
| Coarse run | 3.92 | 3.83 | 2.75 |
| 局部方向窗口 | 20.39 | 16.45 | 8.24 |
| 受限时间视角 | 13.73 | 11.94 | 6.32 |
| 五视角等权分数融合 | 10.39 | 9.82 | 4.26 |

全部18行accuracy/macro-F1见artifacts/metrics.csv，逐类accuracy及互补诊断见summary.json。没有只挑最佳系统汇报。

## 能支持和不能支持的结论

1. 视角间存在轻量读出效用差异：windows相对packet在JP +6.09pp、subpage +1.72pp；timing在JP +1.57pp、subpage -0.20pp。只是一个弱读出/seed下的开发结果，不是稳定的原因专属选择规律。
2. 全部释放不必更好：等权融合低于packet。但未做分数校准，存在尺度、相关/重复视角、弱读出误差等解释；不能把它直接解释成需要分级门控的证明。
3. Source packet准确率仅11.57%，最强windows也仅20.39%；这不是成熟WF模型。run序列按索引对齐后取类均值对错位敏感，可能严重低估可学习编码器作用。不能据本轮STOP整个token或burst方向。
4. Exact-run只表示方向RLE，log1p数值转换后保留全部run槽；coarse-only没有精确长度旁路。views维数5000/5000/5000/240/20000，不同偏置和维度意味着并非等容量深网公平对照；窗口是统计读出，不是已训练token网络。
5. 事后oracle union仅示意互补上限，不是部署路由：packet+windows的JP oracle union20.78%（windows16.45%），subpage11.32%（windows8.24%）。使用真实标签得出的上限不可当可部署增益，且弱基线下容易出现表观互补。
6. 本轮没有训练分级机制/学习token embedding、没有TTA/蒸馏、多seed、统计确认或未知条件验证。原型偏好不能证明特征真实不变性。

## 完整性

抽样清单和全行/方向SHA256交叉为0见sampling_and_isolation.json；仅哈希复核（前序已经完整字节审计），不称本轮独立session证明。原型均值/尺度只由source估计。全部18系统预测先写predictions.npz并封存，再评分；source_fit、脚本、PLAN/config均记录hash。

3项新增合成probe测试通过。verify.py独立标量计算指标，检查18行指标、27516条预测、5份封存哈希，0 errors；不重新拟合输入。无共享数据修改。代码保留本run，不修改生产生成器。

## 下一步

保留多视角候选，暂不依据本轮直接设计阈值或按原因释放token。先建立有足够source辨别力的可学习读出基线，并匹配输入/训练/选择预算；重点检验窗口、方向、受限时间的互补和run序列学习是否改善。GPU资源与训练预算需明确，不在本轮自动续训。若做分级机制，需要在无目标标签条件下识别何时切换的信号，不能用JP/subpage真实条件身份当部署oracle。
