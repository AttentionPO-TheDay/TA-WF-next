# n-shot 冻结表征维护 v17

用户明确允许目标域n-shot真标签，并授权开始训练。纯净源骨干、BN参数与缓冲区全部冻结，源训练清单仍为18553行/102类。仅TemporalDrift Day90/270，沿用已有1024 reference与独立query，不重划分；这些日期已观察，只作机制开发。WTT/AWF关闭。

## 覆盖审计与训练前修订

初始结构审计未训练/评分，Day90最少1条/类、Day270最少2条/类。因此原拟严格3/5-shot均记不可行；严格1-shot为主分析。训练前明确加入每类至多3/5条补充预算曲线，取min(n,reference类计数)，不称完整3/5-shot，不减类、不从query补齐。修订依据只有reference覆盖，没有query评分反馈。覆盖原记录保留在coverage_audit.json。

reference1024标签作为类别索引用于标准平衡n-shot基准采样与覆盖审计；仅被选support真标签可参与优化。这不是未知类别流上仅花n*C次标注的主动采集证明。三model seed42/100/2026×三support seed1701/1702/1703×两日期；抽样按原始内容哈希嵌套且跨modelseed共享。每类1条为102标签；cap3/5实际预算与缺额逐项报告，各arm完全相同。

## 方法与对照

新run显式复制纯净源模型实现，三checkpoint按源审计SHA核对。通过classifier输入hook提取实际分类特征，独立样本eval；不拟合query统计。源每类固定24条（总2448）仅用于特征标准化与源类原型。零标签对照为原frozen和源原型读出。

目标原型：源特征均值/标准差变换（std下限1e-3）后逐样本L2归一，支持类均值再归一，余弦最近类输出。收缩原型主臂：目标类均值按实际n与源类均值按固定伪计数3加权，最后L2归一；不使用Teacher伪标签。

线性头：在源标准化特征空间，将原分类头代数变换保证初始logits等价；full support CE + lambda/(2*102)倍相对原W/b平方和，lambda=1，Adam lr=.01，固定200步；只训练头，不接入LoRA，无额外目标验证集、早停或选最佳epoch。三方法×54支持状态共162个拟合状态，其中54个监督线性头。小模型CPU训练；GPU2仅一次性冻结特征提取，batch128（与历史frozen一致）、进程显存限8%、CPU2线程、总wall<=3600秒、输出<=2GiB，不中断已有GPU任务。

## 评价与判决

所有拟合完成并冻结hash后才进入query标签终点评价。报告固定102类Macro-F1、Accuracy、相对原frozen的ΔF1、NAR=原对变错/原对、Recovery=原错变对/原错、R/D/W计数。输出均直接采用固定模型，不以query选阈值部署。

另以归一logit候选对原类别的margin gain对改写排序，绘制query标签定位的诊断NAR/Recovery前沿，NAR分母为全部原正确。相同实际D预算下比较Recovery；匹配纠错数比较D/W，达不到记不可行。主比较为严格1shot shrink_proto vs target_proto，每日期平均F1至少+.5pp且相同D预算Recovery至少+.5pp、按modelseed聚合至少2/3不下降；这是开发诊断门槛，不是部署风险保证。cap3/5及head为预定次级，不据结果替换主臂。

不把support抽样×模型seed视为独立数据集。报告两层均值/波动、逐cell结果；同support不同方法成对比较，不能仅超过frozen宣称机制优势。不把1shot用于监督安全概率校准或声称NAR<=1%保证。

## 审计与停止

训练前冻结代码/config/PLAN、checkpoint、prepared数据、固定split及历史基线hash；核对support/query/source原始timestamp内容隔离和类覆盖。提取后原骨干state不变、logits与历史原冻结预测同argmax且误差<=配置容差。拟合API仅接收support特征和选中标签，query不参与拟合；保存全部支持清单、模型、预测、进度、成本。终端独立重载所有小模型、复算全部预测/指标/前沿/支持标签数量，GPU重提每状态前32条特征。失败保留并登记，禁止因阴性追加调参。

## 训练前实现修复 v17.1

首次冻结特征提取batch16在seed42/Day270的历史argmax严格复现断言失败，logits容差检查已过；停止时尚无任何support拟合或新query标签指标。原因候选为batch大小改变的浮点近边界差异。原配置/冻结清单/代码与部分特征、失败日志封存于artifacts/failure_before_training_batch16。改用历史原batch128，不放宽logit容差或Top1一致要求；审计重提使用完整原batch后比较其中32行，避免审计自身改变batch形状。标签/支持清单/方法/优化参数不变，重新冻结后重跑特征。本修复不是基于目标性能返调。
