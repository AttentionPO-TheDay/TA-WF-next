# 实验结果

完成：4种长度编码×3seed，固定source2040/valid510、15epochs。CPU4线程，GPU隐藏，耗时2分20.77秒，峰值RSS2263216 KiB（约2.16 GiB）。不读取未来日期/WTT/AWF。

## 开发集结果

均值±样本标准差，单位%；选模指标valid macro-F1（平局最早）。

| 长度专属编码 | accuracy | macro-F1 | 参数 |
|---|---:|---:|---:|
| continuous：连续log值线性投影 | 5.69±0.34 | 2.47±0.34 | 79094 |
| bucket：对数分桶embedding | 22.22±1.88 | 20.39±2.13 | 79286 |
| hybrid：分桶embedding+连续投影 | 22.35±2.07 | 19.63±2.64 | 79318 |
| hybrid_norm：组合编码+融合后归一化 | 21.50±3.19 | 18.96±3.50 | 79382 |

主比较hybrid-continuous平均accuracy +16.67pp、macro-F1 +17.17pp；三个seed两指标均正，通过预定筛查门槛。两者具有相同精确长度信息可用性，bucket是其确定性派生。支持本预算下长度编码对学习有实质影响，不把exact弱表现归因于精确信息无用；参数差较小，但不是严格等参数或完整表达能力匹配实验。

hybrid-bucket平均accuracy仅+0.13pp，macro-F1 -0.75pp；macro-F1三seed差为-1.55/+0.15/-0.85pp，未通过一致收益门槛。不能证明连续精确长度分支在分桶之上带来稳定增量。hybrid_norm-hybrid平均accuracy -0.85pp、macro-F1 -0.68pp，无支持增加归一化的证据。不得只报最高accuracy条件而忽略选模主指标。

## 解释与后续

这次专属层是方向embedding、长度类型感知编码、位置编码与保序CNN的组合，未加入独立window分支、边界分支、TLS flow层级、资源标签或蒸馏。只借鉴CipherSight数值双编码思想，不是论文复现。其训练期特权信息机制不能由当前结果代替。

当前建议保留bucket作为简洁token编码基线；不采用尚无增量证据的hybrid/norm作为默认。后续优先做容量受控的run/window专属分支与融合消融，判断两种表示是否互补；须新run冻结后执行，本轮未自动扩展训练。生成器的漂移职责不变，训练期token分支最终如何向部署模型转移仍需单独蒸馏验证。

验证集已经用于多轮开发与checkpoint选择，不是独立测试；重复单位仅训练seed，只有3个，不作统计显著性或抗漂移结论。有限15epoch未证明收敛。训练与评价均保留token输入，本轮不证明仅训练时使用token就能改善packet部署模型。公共模块按配对seed初始化（与上轮初始化顺序不同），因此本轮bucket22.22%与历史23.01%的差异不是性能退化的受控证据。

## 完整性与版本

12指标、6120预测独立重算，全部checkpoint重载预测一致，全部epoch选模核验通过。输入缓存hash匹配历史封存；当前生成器逐条重建全部2550样本的方向、连续长度、分桶及padding，与缓存一致；原始方向hash与抽样清单一致，source/valid交叉为0。padding无影响、分支梯度、输入权限及公共初始化测试通过；仓库36项测试执行，35通过1跳过。

实际配置见config.json；来源为本项目20260922T060351Z_cpu_token_native_classifier_6463ceb7的固定token与抽样清单（没有加载其checkpoint），原数据根通过configs/datasets.json解析。训练前PLAN/config/train/verify/生成器/数据配置及缓存hash见artifacts/pretraining_seal.json；输出hash见output_seal.json；逐seed结果、配对差和核验见aggregate.json、integrity.json。代码显式重新实现前轮小模型流程，不依赖旧目录训练代码；旧虚拟环境仅提供依赖。论文参考：https://arxiv.org/html/2608.13905v1 。
