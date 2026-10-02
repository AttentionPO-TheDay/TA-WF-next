# 四模型原生实现审计结果

状态completed：DF、Var-CNN、RF、ARES四模型并行源码审计完成。0训练、0新真实数据/标签读取、0未来评分；下载并归档公开源代码，复用历史已观察指标。

|模型|输入|关键发现|决定|
|---|---|---|---|
|Var-CNN官方dir-only|前5000包方向|作者代码支持；本地旧派生实现的因果padding、shortcut、SAME池化及分类头不一致|第一优先：显式按固定源码移植后运行|
|RF作者原生|同5000包方向＋时间TAM|不是direction-only；作者Adam有wd.001及指数LR衰减，Proteus配方没有|第二优先：source/valid时间及TAM接口审计后单列原生输入比较|
|DF作者结构移植|前5000包方向|现有参照有明确来源，框架差异已标注；accuracy/F1选模在现有曲线选中同一ckpt|复用历史accuracy70.784%、F1 69.643%，不重复同配置|
|ARES Proteus端口|前10000包方向|硬编码32token位置及top20；5000包不兼容；eval仍随机roll|暂缓，先解决来源、输入权限及确定性问题|

当前CNN＋MLP＋局部遮挡历史accuracy73.987%、F1 73.394%仍为开发参照。论文高分不能保证本项目source150/valid510达到90%。Proteus正文声明官方代码、调参及5折CV，其图表的未来日期F1不是当前源期accuracy；样本预算和选择方式不可混称一致。

## 推荐执行顺序

优先准备Var-CNN官方dir-only移植，另准备RF源期时间接口审计。两者可并行准备，但分开报告输入权限；RF胜过方向CNN也不证明纯网络更强。DF保持历史参照，ARES暂不占训练预算。具体源码、缺口和配方见DF_AUDIT.md、VARCNN_AUDIT.md、RF_AUDIT.md、ARES_AUDIT.md。

后续有界方案见NEXT_EXPERIMENT.md：统一三seed21729/23407/22026与固定source150/valid510，新运行accuracy选模；Var-CNN原生最多150次验证，RF适配20次，历史结果不能声称选模预算匹配。原生流程比较与机制公平归因区分记录。计划仍draft，本轮没有启动训练或安装过时依赖。

## 来源与核验

DF固定38df0c15a089f13228e4df06bd5269c8a37340fa已有快照；Var-CNN固定e5db76a86fcbf8839f764b9aca241749d1c1f700，9份公开文本经GitHub connector获取（URL、blob及本地SHA均保存）；RF固定b75680148429c5f554114b78a0f846541b39f0ec，公开GitHub源码下载与SHA；Proteus固定4cdab4163bf3de7036a2498dac533c888b97664d归档源码明确复用，不导入执行。ARES审计静态形状/资源估算不是已完成GPU运行验证。

所有审计报告及产物SHA记录于artifacts/audit_manifest.json；本轮新增脚本只有公开源码下载，不存在训练执行入口。后续freeze前仍须完成各模型移植及输入适配的行为验证。
