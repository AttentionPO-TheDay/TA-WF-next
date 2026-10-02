# 生成器与分类器归因诊断

状态：frozen；用户授权执行，所有探针和裁决规则已于新评分前冻结。新实验唯一run，复用上一轮C/B/D三个条件各三seed的best checkpoint；不重训完整模型，不重选checkpoint、不修改已有结论。

## 数据、模型与权限

固定TemporalDrift source2040/valid510、102类和5000包方向，复用已审计prepared.pt及sampling manifest，按configs/datasets.json核对来源。不打开未来日期/WTT/AWF或PCAP；无GPU。仅source标签参与闭式ridge拟合，valid仅诊断评分。原best checkpoint已按此valid选择，故本轮不是独立确认；不把新探针称为新的未见测试。不直接导入旧项目训练代码，仅显式复用本项目src生成器/分类器实现与指定run产物。

## 统一探针

两套输入，所有条件同规则：packet_tokens为100×52 token保序展平+100有效mask，共5300维；all_views再附加固定raw run128×4、run mask128、raw window120×4和window mask120，共6540维。后者与原完整模型拥有相同输入信息权限，避免用packet-only与多视角模型直接作能力裁决。不额外做汇聚以免人为压缩掉顺序。

每列仅用source拟合均值/总体标准差（std<1e-6→1），目标为102维one-hot，截距为source类别先验。float64闭式ridge：mean squared error + lambda*||W||²，主lambda=1.0，预先固定0.1/10.0作敏感性检查，全部报告、不挑最佳lambda或据valid改变规则。C/B/D×3seed×2输入×3lambda共54次线性拟合；网络权重完全冻结、eval模式、无梯度。保存特征、标准化、线性权重、预测、混淆矩阵和指标。每个网络的source/valid原预测先重载复算核对原产物。

## 问题与预定判据

主比较B−C，D作慢更新补充。三seed对应比较。方向性门槛沿用平均F1差至少1pp、3/3seed同向；两个固定敏感性lambda的均值差也应同向，否则记不稳健。

1. B的packet探针valid比C低≥1pp且满足一致性：支持当前线性读出下生成器表示退化。只有当source探针F1不降、source→valid落差更大，才进一步称为生成器表示过拟合模式；不能把训练/验证同时下降称为孤立过拟合。
2. B的all_views探针比其原完整模型valid高≥1pp、三seed为正且敏感性同向：支持现有分类器尚未利用全部可线性提取的泛化信号。该差异也包含读出正则化/优化方式差异，不等于证明某一层实现错误。
3. B的packet探针比C更好而完整模型B比C差：支持生成器信号与下游利用之间发生反转。
4. 两类证据可以同时成立；结果也可能只支持表示退化、只支持读出不足或证据不足。不能为了用户提出的二选一强行归因。

## 轨迹与类别诊断

复用20个每5epoch评估记录，绘制source accuracy、valid F1及source诊断token变化，报告55..100轮的描述性相关和best→100变化。不将相关当因果，训练时间是共同混杂。仅best/latest权重被保存，不能伪造中间epoch的token探针曲线。

对原分类器和主lambda两套探针，保存102类召回/混淆及B−C逐类变化；报告改善/退化/持平类别数和主要混淆对。三个seed是在同一valid上的模型重复，不当作新增1530条独立样本，不从少数有利类别推整体。

## 预算与核验

3个CPU worker各2线程，最多6线程；每job360秒、整个执行1800秒上限；单进程RSS6GiB。9个checkpoint job，不追加条件、随机seed或主网络训练。独立核验采用保存的特征和source统计重算预测/指标、检查ridge正规方程残差、记录并核对原始标签/输入/checkpoint散列。预定54探针×source/valid=108组新预测，另18组原分类器预测重载；54组权重/标准化核验。保存完成结果；预算/核验失败明确不完整。

适用限制：固定线性探针不是表示能力上限；标准化与L2受坐标变换影响，敏感性检查也不能完全消除此问题；既有模型共同训练导致共适应，无法单次证明唯一因果。结论必须区分观察事实、诊断支持与尚待验证的原因。

执行前检查：合成数据上dual/primal ridge解与预测一致；常量列处理通过。九个checkpoint及原指标/预测/日志散列、固定行索引、标签计数、source/valid方向交叉0、原实验完整核验均通过。准备阶段未产生新真实性能分数。实现只导入本项目src；显式复用/home/rbf/TA-WF/.venv环境，不导入旧训练代码。

## v2实现修正（2026-09-26）

v1共8/9 checkpoint job完成、48个探针已评分，D2026在原预测复算时失败，未完成归因。审计见artifacts/v1_invalid/replay_audit.json：仅D2026 source行864，在全部参数requires_grad=False时top2浮点舍入为并列，argmax由原71变43；保留原requires_grad标志则source/valid均精确复现，分开调用generator/classifier与原forward无影响，valid原预测全部一致。

修正仅保留原参数标志；全部前向仍在torch.inference_mode下，无梯度或optimizer，不更新网络。全部9组统一重新提取/拟合/核验，v1全部作废保留。lambda、输入、数据、checkpoint、裁决规则均不改变。v1模型产物和源代码在artifacts/v1_invalid，日志在logs/v1_invalid。

执行记录补充：首次修复控制脚本发生Python语法错误，随后的串行shell命令未及时中止，发生一次14.18秒的启动失败；所有worker在已有输出目录检查处退出，未额外拟合。该次尝试覆盖了顶层pipeline记录、部分日志和冻结时间戳。原8组probe产物及其记录的配置哈希保持不变，归档配置保留相同统计设置但时间戳已更新，不能声称是首版逐字节配置副本；参数开关不一致原因由独立replay_audit复现。

预算保守预扣150秒覆盖首版53.72秒、数值审计和14.18秒失败启动；每job预扣30秒，v2上限330秒。有效探针仍54组，实际包含作废48组总102组闭式拟合；不称独立重复或新增方法搜索。
