# 固定生成器后的独立分类器重训

用户授权：开始上一轮建议的下一步实验。CPU，6个训练job，不追加条件或搜索。

## 问题与对照

显式复用20260926T040212Z_classifier_generator_parallel_946c81c4的B联合训练生成器、C随机冻结生成器，各1729/3407/2026的best checkpoint。仅使用其中generator权重；原classifier只用于预测重放核验和历史指标对照，绝不作为新分类器初始化。生成器固定eval，以无梯度方式缓存100×52 packet token；保留原run/window、mask、span，无新标准化。新classifier与原HierarchicalViewTransformer同结构113514参数，packet投影Identity。每对B/C按原seed+100000创建相同初始classifier；相同batch顺序与dropout随机流，无交叉加载不匹配的分类器。

## 数据和选择权限

沿用已审计TemporalDrift source2040/valid510、102类、每类20/5条、前5000包方向；数据路径由configs/datasets.json定位，prepared.pt与manifest及全部锚点散列见config。不重分数据，不打开未来日期、WTT/AWF或PCAP。source标签用于交叉熵训练；valid只按每5轮一次、最高Macro-F1且并列取最早的规则选checkpoint。两种生成器曾在同一valid上选模，故这是已观察开发数据上的诊断，不是独立确认；B在生成器阶段已使用额外监督优化，当前只匹配分类器重训预算，不称总训练算力匹配。

## 冻结训练方案

每job100epochs，batch64，AdamW lr0.001/weight_decay0.0001，dropout0.1，无scheduler，与原分类器相同。评估epoch5..100，共20次选择机会。CPU6worker×2线程，最多12线程；每job含缓存最多3600秒、验证300秒、全pipeline4200秒，RSS每worker4GiB/总24GiB。监督器超时/超内存停止并保留部分产物，不自动重启或延长预算。best与latest、优化器/RNG/逐轮历史、缓存token、预测与核验记录均放本run。显式复用旧项目虚拟环境，不导入旧项目代码。

## 预定比较与解释

主比较：新B分类器−新C分类器valid Macro-F1，报告三seed逐项与均值及accuracy；均值≥+1pp、三seed均正且accuracy均值不降，记固定B表示优势；均值≤−1pp、三seed均负且accuracy均值不升，记固定B表示劣势；否则未决。

恢复比较：新B−原B同样要求平均F1≥+1pp、三seed均正、accuracy均值不降。仅当固定B表示优势与B恢复同时满足，而原B−原C为负，才支持“联合训练/共适应限制了已有表示的利用”。若固定B表示劣势成立，且source F1差非负、source-valid差距增加，支持“表示训练集偏向”，仍不能证明唯一因果。若B与C重训同时提升，须报告两者恢复和差分中的差分，不能全部归因于B解耦。任何门槛不通过均保留未决，不调超参救结论。

本轮使用与原模型同结构读出，无法排除其他分类器结构更好；三seed共享510条valid，不当作独立1530样本。原始性能为历史参考，不作为新增独立重复。结果只针对源期分类能力，不涉及漂移、预训练或TTA。

## 执行与核验

执行前检查配对初始化、合成输入cache与在线logit等价、classifier能更新且generator无梯度/无更新；正式缓存先复现原12组source/valid预测，并对照上一轮已核验缓存特征。保留原requires_grad标志进行无梯度推理以避免上一轮CPU数值路径问题。分类器训练在生成器之外进行，缓存转为普通无梯度tensor。完成后另进程重载6个最佳classifier、重算12组预测与指标，核对选模历史、初始权重配对、生成器与缓存散列及数据权限。错误或不完整不得称实验通过。

冻结前记录：合成接口检查首次因测试fixture缺span字段而退出，补全fixture后全部检查通过；未执行真实训练或新验证评分。冻结完成时间见config.json；正式运行以artifacts/freeze.json的代码/配置散列为准。
