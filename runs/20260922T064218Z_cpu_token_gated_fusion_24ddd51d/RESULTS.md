# 实验结果

完成：五条件×三seed、source2040/valid510、15epochs，CPU3线程，GPU隐藏。墙钟4分06.52秒，峰值RSS2263652 KiB（约2.16 GiB）。未来/WTT/AWF未访问。

## 固定预算开发结果

单位%，均值±样本SD；valid macro-F1选模，平局最早。

| 条件 | Accuracy | Macro-F1 | 参数 |
|---|---:|---:|---:|
| bucket | 22.09±1.77 | 20.04±2.36 | 79286 |
| hybrid直接相加 | 22.29±2.01 | 19.61±2.52 | 79318 |
| separate独立残差MLP后相加 | 22.75±2.18 | 20.52±2.18 | 79862 |
| separate_fixed连续分支固定乘0.1 | 23.27±1.38 | 21.26±1.35 | 79862 |
| separate_gate学习门控 | 23.01±1.33 | 20.98±1.43 | 79895 |

主比较gate-hybrid：F1差+1.41/+0.22/+2.46pp，均值+1.37pp；accuracy均值+0.72pp，通过预定开发筛查。gate-bucket：F1差+0.15/+0.30/+2.36pp，均值+0.94pp；accuracy均值+0.92pp，亦通过。不是显著性或独立确认。

关键归因：gate-fixed F1差-0.65/-0.10/-0.10pp，均值-0.28pp；accuracy均值-0.26pp。三个seed均不如固定衰减，不能将gate相对hybrid的改善归功于学习门控。separate-hybrid F1差+1.33/+0.42/+0.99pp，均值+0.91pp；accuracy均值+0.46pp，通过筛查，支持本实现的独立处理候选。fixed-separate F1差+0.73/-0.10/+1.57pp，均值+0.73pp，但有一负，未通过一致性门槛；固定衰减均值最高不等于已证优于所有条件。

门控确实学习而非保持常数：所选epoch固定source前128样本有效token的gate均值分别0.336/0.103/0.189，取值范围约0.041–0.616。这里只是source子样本诊断，gate数值不等于特征重要性；不能据此选gate阈值或推断漂移路由。

## 与并行诊断的关系

独立诊断run 20260922T064229Z_cpu_token_branch_diagnostic_aa17da3e使用上轮checkpoint：full/bias_only/off valid F1为19.63/16.98/7.94%，删除连续变化部分三seed均下降。连续变量RMS并未压过bucket。因此不支持“连续分支只是有害噪声或幅度压制，推理关掉就更好”。诊断是固定模型依赖检查，移除分支会造成训练外激活分布；新训练则比较重新联合优化，二者不矛盾。两项方案各自事前冻结，新训练未据诊断结果变更。

## 结论及剩余问题

支持把逐token独立处理+受控融合继续作为候选；暂不需要学习门控，固定0.1版本可作简洁候选。精确长度是否独立贡献仍未隔离：本轮缺少“仅bucket也添加同一残差MLP”的处理匹配对照。后续应先补该对照及固定融合下连续分支的有界消融，区分额外非线性/优化收益与精确值收益，而不是立即加window或宣称生成器已获最终验证。此建议未自动开始新训练。

参数差小于0.77%，不是严格等容量；新增模块是逐token MLP，不是独立序列网络。所有模型仍是小样本15epoch，6/15所选epoch为末轮，未证明收敛。三seed和valid多次开发不足以作显著性、抗漂移或外部确认结论；训练/评价都保留token输入，尚未证明训练专用蒸馏。为控制并行总线程，本轮3线程而上轮4线程，浮点归约及训练轨迹可能变化，baseline数值略不同；所有主比较均使用本轮同线程对照，历史基线不算独立新证据。

## 核验及版本

15指标、7650预测独立重算，全部checkpoint重载预测一致、选模/封存/输入隔离核验通过；2550样本由当前生成器全量重建字段与历史缓存一致。padding、分支含gate梯度、公共初始化、fixed/gate初始输出等价及参数预算测试通过；全仓库35测试通过1跳过。日志见logs/train.log。

配置与代码hash见artifacts/pretraining_seal.json，输出hash见output_seal.json，全部逐seed和配对结果见aggregate.json、history.json，核验见integrity.json。实现显式复制本项目20260922T062732Z_cpu_token_typed_encoding_df081d9e脚本后加入独立MLP/门控；缓存及清单来自本项目20260922T060351Z_cpu_token_native_classifier_6463ceb7；无旧目录checkpoint/训练代码依赖，旧虚拟环境仅提供依赖。数据根由configs/datasets.json定位。
