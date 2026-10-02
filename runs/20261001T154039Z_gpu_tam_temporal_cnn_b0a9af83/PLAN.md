# 用户授权CPU转GPU：保序卷积分类器对照

用户明确要求剩余CPU实验切换GPU。原并行run在18项GPU和1项CPU新训练完成时停止CPU队列，部分CPU模型/checkpoint与日志全部保留。直接跨设备续训会改变数值和随机数路径，不能当原CPU控制实验完全延续。因此独立登记本run，在GPU0从头训练同卷积设计3seed，以已完成GPU Transformer基线为同设备对照。不重新启动原CPU baseline，不把已完成CPU CNN单seed75.294%混入GPU三seed均值。此前六GPU方向与CPU单seedvalid已观察，本run是有反馈的开发，不能称新确认。

## 数据和权限

只用原固定TemporalDrift source102×150=15300、valid102×5=510，configs/datasets.json为来源，prepared.pt字节复制前run缓存并校验一致。输入仅方向条件TAM2×1800，按原规则80秒及以上归最后bin、log1p。仍保留缓存前5000有符号时间戳，但packet槽位全mask，timestamp改变不影响模型输出；不排序/差分/新重建TAM。无原始NPZ新读取、重划分、未来日期、WTT/AWF、teacher/预训练/适应。source CE梯度及source拟合诊断，valid只20次固定选模/评价、不进梯度。旧checkpoint只baseline重载审计，GPU新模型不加载CPU或旧best权重。

## 模型与同设备对照

temporal_cnn与前run cpu_temporal_cnn结构完全相同，只改变条件名字和运行设备。固定多尺度生成器两个kernel5 Conv2→16→16、GELU、共享dilation1/3/9并等权同位置融合；每token完整30维原计数和80维有序子区间学习特征，120×110时间token、全有效时间mask，100packet槽位全mask且packet参数冻结。固定time projection110→128、时间位置/类型embedding、final norm、均值读出256→102。

分类器只有两个保序时序残差块，每块LayerNorm128、kernel3 Conv128→128、GELU/dropout0.1、kernel3 Conv128→128/dropout0.1和skip。对照baseline为两层pre-norm Transformer d128/4heads/FF256/dropout0.1，原历史GPU基线accuracy75.948%、F1 75.295%。两模型共享模块同seed初始state一致，head不同、参数和计算量不同；不称纯attention开关或算力匹配。

## 训练、选模和预算

seeds21729/23407/22026，从头初始化；沿用source randperm seed+9000+cycle至12800×64索引流，与GPU历史baseline逐字一致。batch64、12800步、CE、AdamW lr0.001/wd0.0001，1–6400步lr0.001、6401–9600步0.0003、9601–12800步0.0001，无增强、AMP或TF32，确定性算法，每任务2CPU threads。全部20次source/valid评价steps3680+480*i、i0..19，eval batch64；valid accuracy严格提高选best、平分最早、不early-stop。不受前run mask正向结果影响添加增强。

仅GPU0三任务并行，GPU1/2现有占用不使用；每任务3600秒、总管另120秒启动/重载watchdog，整批7200秒上限，不加seed/步数/调参。原CPU步骤不算本run新重复，原CPU已消耗算力如实在旧run保留；本次新增预算是用户授权设备切换下的三任务重训，不隐藏为无损恢复。

合成7项检查复用前run并重新运行；GPU真实source64×3模型前后向和显存预检、共享模块初始化、历史GPU baseline三checkpoint共6组source/valid重载预测核验完成，无optimizer update。聚合显存估计含三数据副本、三个CUDA上下文、optimizer缓存与4GiBreserve，小于24GB实际显存；启动后核对实测。

训练前冻结本run方案/config/code/source-cache/预检以及6份初始state/索引/历史预测/曲线等hash。入口拒绝draft；首步active generator梯度有限非零、frozen无梯度；best generator参数更新/frozen不变，保存初始化/索引/完整曲线/best和latest/预测/logits，重载新实例核验全部source/valid预测及独立accuracy/F1。总管核验6报告×best/last×source/valid共24组指标、索引、共享初始化、步数/20次选模和最早best。非有限值、hash/权限/重载/梯度/参数检查或预算失败停止并保留产物，未获授权不扩大预算。

## 裁决与限制

主指标三seed平均valid accuracy；逐seed与Macro-F1、source拟合、best/last、参数/耗时报告。temporal_cnn−baseline平均accuracy≥1pp、3/3seed差>0、平均Macro-F1不降才过开发门槛；平均accuracy≥90%独立判断。前run mask77.647%只为明确的历史开发候选，不作为不同设备混合平均或偷偷加mask。若CNN未过，保留前run已通过的mask候选，不自动拼接结构或扫描新超参数。

同510 valid反复开发，三seed不是三个独立数据集；设备切换经用户授权但不消除已观察结果。CPU与GPU路径不能认定数值等价，故本run仅同GPU对照；新GPU重训不是新样本确认，也不能单凭源期收益证明外部泛化或抗时间漂移。源码来源与保留记录见SOURCES.md及原run设备切换登记。
