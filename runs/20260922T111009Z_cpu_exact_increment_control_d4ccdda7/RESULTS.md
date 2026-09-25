# 实验结果

完成：四条件×三seed，source2040/valid510原清单、15epochs，CPU3线程，GPU隐藏，无未来/WTT/AWF数据访问。

## 结果

单位%，均值±样本SD；valid macro-F1选模。

| 条件 | Accuracy | Macro-F1 | 参数 |
|---|---:|---:|---:|
| bucket_mlp：仅分桶+专属MLP | 22.94±0.98 | 20.87±0.81 | 79558 |
| fixed_bias：加常量输入分支 | 23.07±1.45 | 21.04±1.43 | 79862 |
| fixed_bucket：同结构分支仅用桶下界 | 23.14±1.22 | 21.19±1.25 | 79862 |
| fixed_exact：同结构分支用精确长度 | 23.27±1.38 | 21.26±1.35 | 79862 |

主比较fixed_exact-fixed_bucket：F1逐seed差+0.272/+0.120/-0.179pp，平均仅+0.071pp；accuracy平均+0.131pp。未通过三seed一致门槛，没有精确长度稳定独立增益证据，也不是统计等价或精确长度无用的证明。

fixed_exact-bucket_mlp：F1差+0.308/+0.966/-0.107pp，平均+0.389pp；accuracy+0.327pp，亦未通过。fixed_exact-fixed_bias F1平均+0.220pp、两正一负，未通过。fixed_bucket-bucket_mlp F1三seed均正，平均+0.318pp、accuracy+0.196pp，通过开发筛查，但幅度小、最小seed仅+0.036pp，不作统计显著性主张。bias-bucket_mlp F1平均+0.169pp、两负一正，未通过。

## 对前轮结论的更新

前轮固定融合候选优于简单分桶，不能据此把收益归因于新增精确值。本轮为桶加入相同专属处理后差距明显缩小；进一步同参数、同处理、仅桶代理的分支已经接近真实精确值。结果更支持优先研究表示的处理/优化方式，而不是声称精确信息已被证明有独立增益。原先“模型用到了连续信息”的训练后关闭诊断仍成立：共同训练模型依赖一个分支，不代表该信息在重新训练的匹配基线之外有不可替代价值。

建议暂保留仅桶+专属MLP作为简洁主基线，桶代理分支作为小幅改善候选，精确分支不默认加入。不要继续围绕同一valid微调0.1或挑seed追逐极小差异。下一步若继续，优先扩大预定训练预算检查收敛（所有对照同预算），或转向有界run/window互补性实验；另立run冻结，本轮未自动扩展。不能据此舍弃生成器、多视角设计或宣称最终性能。

## 限制和核验

只有3个训练seed、每类source20/valid5，valid已多轮用于开发且选checkpoint，不是独立测试。有限15epochs不能证明收敛；桶代理使用预定桶下界，非所有coarse-only映射。固定比例继承前轮开发选择；fixed_bias连续权重无数据梯度，名义同参数不等于有效容量完全相同。训练/评价保留token，未验证训练专用蒸馏或漂移机制。

12指标、6120预测独立核验、选模、checkpoint重载、封存及source/valid隔离通过；2550样本全量生成器重建字段与封存缓存一致。三个非exact条件更改精确输入不影响输出，exact对精确值敏感；exact在桶代理输入下与fixed_bucket初始输出一致、参数完全相同；公共初始化、padding、梯度检查通过。全仓库35测试通过1跳过。fixed_exact与前轮separate_fixed的全部1530 valid预测逐条复现，此锚点复跑不当独立新证据。

配置、代码、输入版本见artifacts/pretraining_seal.json；输出hash见output_seal.json；全部逐seed、配对差见aggregate.json；核验见integrity.json；耗时和资源见logs/train.log。显式复制本项目20260922T064218Z_cpu_token_gated_fusion_24ddd51d脚本后加入匹配信息对照；原token清单来源20260922T060351Z_cpu_token_native_classifier_6463ceb7，路径由configs/datasets.json定位。无旧目录训练代码/checkpoint依赖，仅使用旧虚拟环境依赖。
