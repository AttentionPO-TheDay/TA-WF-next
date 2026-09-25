# 20260922T064218Z_cpu_token_gated_fusion_24ddd51d

问题：独立长度分支和学习门控能否改善直接相加的macro-F1

状态：frozen，训练前冻结，独立于并行诊断结果；用户授权CPU验证。

样本：沿用本项目token-native run的source2040/valid510（已隔离的TemporalDrift源数据），复用封存token及清单，不重新划分。原数据通过configs/datasets.json定位。source标签梯度训练；valid逐轮macro-F1选checkpoint，平局最早；未来日期/WTT/AWF关闭，不适应。valid为多次观察的开发集，非独立确认。

固定输入：前5000观察的前512run，方向、长度桶floor(log2(count))+1、log1p(count)/log(5001)，padding mask。不添加窗口、时间、边界、资源标签。训练与评价都保留token层，训练期专用蒸馏不在本轮范围。

五条件：bucket=仅分桶embedding；hybrid=桶embedding+连续Linear直接相加；separate=桶和连续16维向量各自经过独立残差MLP x+GELU(Linear16-16)再相加；separate_fixed=同上但连续分支固定乘0.1；separate_gate=同上但连续分支由sigmoid(Linear32-1([桶,连续]))逐token门控。门控权重初始0、bias log(0.1/0.9)，使初始输出与fixed一致。fixed对照区分简单衰减与可学习门控。门控不是漂移路由器，无类别/日期真值输入。

其余均为方向Embedding3-16、位置Embedding512-32、两层32通道k5卷积、16段masked pooling、102类分类器。公共模块配对初始化；新独立MLP配对初始化。参数差<1%但非严格等容量；MLP为逐token专属处理，尚非各分支独立序列网络。实现显式复制上轮20260922T062732Z_cpu_token_typed_encoding_df081d9e/train.py与verify.py并最小修改，不运行旧目录训练代码、不加载旧checkpoint。两个baseline从头复跑作本轮同预算锚点，不算全新独立证据。

预算：5条件×seed1729/3407/2026，各15epochs、batch128，AdamW lr0.001 wd0.0001，CPU3线程+并行诊断1线程，总GPU隐藏；训练timeout600秒。非有限/hash错/隔离错/超预算停止，禁止根据并行诊断或valid结果变更本轮超参、条件或预算。

主比较gate-hybrid；只有三个seed macro-F1差均>0且平均accuracy差>0才通过开发筛查。gate-bucket同样门槛评价实际基线收益；gate-fixed隔离学习门控增益；separate-hybrid与fixed-separate为机制消融。全部报告，不将最佳单seed当结论。记录accuracy、macro-F1、train loss/accuracy、source固定前128样本有效token gate统计、参数、耗时。gate值不是特征重要性的充分证明。

验证：生成器全输入重建、hash、隔离、padding、参数梯度（含门控）、公共初始化、fixed/gate初始等价、checkpoint预测重载、全指标独立重算及选模核验。保存PLAN/config/脚本及输入封存、历史、预测、checkpoint。只作此有限训练预算的开发证据，不证明收敛、抗漂移或最终模型效用。
