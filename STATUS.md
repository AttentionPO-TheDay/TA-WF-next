# 当前进度

当前候选设想（2026-09-22，用户确认）：研究分级释放不同特征token的机制，面向不同漂移情况考察不同视角的效用。尚未实现分级选择器，未证明能从部署可观测信息识别适合释放的视角；不得默认使用真实漂移原因/目标类别作为路由输入。现有多视角生成器只是候选特征接口，不是已验证适应机制。本次仓库整理仅本地Git提交，不训练或推送远端。

当前实现事项（2026-09-22）：多视角表示生成器v0已完成，run `20260922T005618Z_multiview_feature_generator_0b5365cf`。traffic_views.py独立输出packet方向、exact/coarse run、局部方向窗口、显式可选同方向时间/run跨度、signed-size视角；字段边界见TRAFFIC_VIEWS.md。32项测试31通过1跳过，三场景384行真实接口检查通过。未训练评分、未读真实标签/URL、未开放WTT/AWF。尚未实现时间bin、batch collator、模型融合/蒸馏；下步需冻结视角对照和数据角色，不能把接口通过当抗漂移证据。下列旧实验结论保留。

状态：共享更新干扰存在性诊断 `exp_dce0488c23844cb7` 已完成，预注册裁决 `STOP_NO_USABLE_INTERFERENCE`。固定 archived raw-timestamp DF seed3407 checkpoint 与 Day90 source-statistics Tent，102 个 directed oracle 网站组中 self accuracy 平均 +1.297 pp，但 cross 仅 -0.250 pp、负向 40.2%，mixed 仍 +0.578 pp，cross-damage 与 mixed-cancellation gate 失败；随机对照和幅度审计通过也不能挽救。第二关未执行，未生成 selector/signal/Burst 方法。该结果是 TemporalDrift development evidence，不替 Host 作更大路线决定。WTT-Time/AWF 保持封闭。

已完成：共享数据迁移；独立项目组织；旧研究交接；统一实验记录入口；2026-09-15 更新 PROTOCOL.md v1，确定 TemporalDrift 开发机制、WTT-Time/AWF 评价冻结方法的数据角色与信息权限。

基础模型：已最小迁入 DF、原双分支 VarCNN，并新增方向单分支 VarCNNDirection，位于 src/ta_wf_next/models。来源与接口见该目录 SOURCES.md。迁移时 4 项 CPU 合成输入检查通过。当前 screening 数据适配器与训练评价入口已实现，真实 source 仅用于隔离/长度审计和随机初始化 smoke，尚无训练。显式使用旧环境中的 PyTorch，新包没有旧代码运行时依赖；新项目独立依赖环境尚未安装。

下一项：2026-09-22 第二轮审计 `20260921T185436Z_proteus_identity_isolation_e596ccf3` 完成：23文件475683行视图记录完整内容审计；Version恢复90145条唯一版本锚点，四份drift共有10017条未知身份，048目标可恢复64406条045/046/047样本（各102类）。Network/Behavior除共享源文件外，完整trace及前5000方向无额外交叉；subpage有2661 URL重复副本。抽样736行中459行绝对时间回退，时间token需先核实语义。未训练评分、未建split。建议下一步冻结开发/留出角色并做时间/有效长度审计；共享结构生成器仍为候选。外部WTT/AWF未开放，既往共享干扰阴性结论不变。首轮结构审计与本轮详细证据均保留。

最近完成：`20260921T191053Z_burst_tokenizer_prototype_ef77e595`，共享方向run生成器v0位于src/ta_wf_next/burst_tokens.py，精确与coarse-only视图分离；边界/padding/预算规则见BURST_TOKENS.md。12项新测试通过；全项目21通过1跳过；三场景384条真实source/subpage样本接口检查0错误。未读标签/训练评分，未实现模型adapter或时间分支，稳定性/泛化未验证。下一步冻结开发/留出及表示对照后才考虑训练。此前时间审计结论保留：方向内单调，跨方向gap不可信，采集根因缺转换链；不排序修复。外部WTT/AWF未打开。

旧研究：R3 共享修正候选 STOP；R3-E 删除 donor 位移后保留主要收益。详见 HANDOFF.md。
