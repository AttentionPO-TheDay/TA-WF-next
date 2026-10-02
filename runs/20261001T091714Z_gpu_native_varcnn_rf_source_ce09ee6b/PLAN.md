# 原生Var-CNN方向与RF-TAM源期实验

授权：用户2026-10-01要求“继续发布实验”，承接四模型源码审计与并行比较。解释为启动已审计后的实验，无外部发布行为。目标固定源期valid510的三seed平均accuracy≥90%，不承诺可达。两个模型×seeds21729/23407/22026，共6任务，旧DF/CNN只历史参照不重复训练。

## 数据和信息权限

仅configs/datasets.json定位的TemporalDrift/train.npz、valid.npz固定source150/valid510行。prepare.py读取签名时间戳前5000转float32后核对方向与旧prepared完全一致，source15300每类150、valid510每类5，源期去重/隔离继承已审计manifest。旧venv仅提供第三方依赖，不导入旧项目代码或权重。

Var-CNN只用sign方向5000包，官方dir-only，无metadata/time。RF使用同样5000包方向与abs(timestamp)，按作者TAM规则分1800bin覆盖80秒，>=80秒累积末bin。两者输入信息不同，必须分组解释；RF结果不是纯架构对照。时间审计见RF_PREPARATION.md：非有限/padding及守恒检查通过；混合负delta存在、方向内为零，TAM按单包时间落bin不依赖混合顺序；保持原存储次序不排序。以float32规范化后float64计算bin保证与标量公式一致，可能与直接rawfloat64边界略不同，此输入规则固定。每角色64条逐包作者公式对照、全行计数守恒/标签行号一致通过。

source标签CE训练及诊断；valid只声明的checkpoint选择和Var-CNN验证驱动学习率/早停，不梯度、不适应、不伪标签。新增时间只限源期这两角色，Day14及其他未来、WTT/AWF完全关闭；不读取现成未来TAM。保存native_prepared和哈希，不改共享数据。

## 模型与来源

Var-CNN根据sanjit-bhat/Var-CNN e5db76a86fcbf8839f764b9aca241749d1c1f700官方支持dir-only配置显式移植：因果膨胀ResNet18，stage0投影、SAME池化、直接GAP→102类。保留MIT和来源。初始化匹配He截断分布/stem及头Xavier、BN epsilon1e-5/momentum.01；PyTorch/Keras BN running variance与随机流仍不同，不声称精确TF数值复现。本地旧VarCNNDirection不使用。

RF复制robust-fingerprinting/RF b75680148429c5f554114b78a0f846541b39f0ec最小模型，结构/reshape/BN/ReLU/池化/初始化保持，仅增加RFNative factory及102类适配。使用作者配方而非Proteus删减权重衰减的配方。无新增增强、无预训练/TTA。

## 训练与选择

Var-CNN batch50、最大150epoch、每epoch306步；训练清单按seed+9000一次随机排列后重复（沿用作者已洗牌数据固定generator），不额外留出5%重划分。CE、lr.001、beta.9/.999、epsilon1e-8，显式移植Keras2.0.8 Adam原递推含未bias-corrected sqrt(v)分母eps，非torch默认替代。每epochvalid accuracy；ReduceLROnPlateau factor sqrt(.1)、epsilon1e-4、patience5、minlr1e-5、无cooldown；旧Keras先检查wait再递增，首次连续6次无提升降LR。EarlyStopping patience10同为先检查后加1，连续11次无提升停止。best strict maximum accuracy最早保存；细节见KERAS_RECIPE.md，回调单元验证后冻结。

RF30epoch、batch200、每epoch77步，末batch100保留；每轮seed+9000+epoch-1随机排列，CE、torch Adam5e-4 beta.9/.999 epsilon1e-8 weight_decay.001，lr=.0005*.2**((epoch-1)/30)。预定epoch1..10,12,14,16,18,20,22,24,26,28,30共20次选模，accuracy最大取最早，另报告last对齐作者最终模型规则。本项目best-valid规则是显式适配。

主指标accuracy，Macro-F1同步，90%以全部三个seed均值判断，逐seed/source/last/验证次数/参数/时间/停止原因完整报告。Var-CNN最多150次valid反馈，RF20次，旧历史20次；不能宣称输入/训练/选模预算匹配。回答各冻结原生流程的源期性能；机制归因需要另立同权限同预算对照。历史DF70.784%、CNN73.987%为原规则结果，不修改历史结论。无根据中途结果追加LR/模型或延长预算。

## 资源与核验

GPU0最多2任务并发、Var-CNN最多1实例；GPU1/2不触碰。每worker2CPU线程、FP32确定性、禁AMP/TF32；单任务3600秒、外层60秒退出宽限、整批21600秒上限。显存预检必须覆盖RF batch200与Var batch50及eval；若两者并发预留不满足则在冻结前降至串行，不能OOM后改batch。失败/超时终止并保留产物，不自动重试扩预算。

预检：官方causal conv与SAME pool手算、阶段形状、梯度/重载；Keras Adam手算两步及回调边界；TAM标量与计数检查；两个模型GPU前向/反向、推理、显存。正式训练前冻结所有代码/config/PLAN/输入及原始来源manifest SHA。每任务保存initial state/order、history、best/latest state及优化器/RNG/回调、best/last预测与best logits。新实例重载best逐条argmax一致，sklearn独立accuracy/Macro-F1复算，两角色12份核验。只有supervisor更新STATUS/RESULTS/索引，所有输出本run。
