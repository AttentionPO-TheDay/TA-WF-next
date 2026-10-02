# DF 原生实现审计（2026-10-01）

结论：已有 `DFReference` 可以复用为“官方结构/配方的 PyTorch 移植参照”，不必重复同配置训练。它不是原 Keras 数值复现，也不是本项目更早的 `src/ta_wf_next/models/df.py`。本轮只读源码和已有报告，没有训练、加载模型或读取新增数据。当前目标是固定源期 valid510 上三 seed 平均 accuracy ≥90%；不能由 DF 论文不同数据结果推断可达。

## 证据定位

下述路径均相对 `/home/rbf/TA-WF-next`；行号取本次审计时文件。为缩短引用，定义：

- `R = runs/20261001T041301Z_gpu_source_fit_native_baseline_ee3c7640`
- `U = R/artifacts/upstream`
- 原仓库 `https://github.com/deep-fingerprinting/df`，固定 commit `38df0c15a089f13228e4df06bd5269c8a37340fa`（`U/revision.json:2`）。本次不访问网络、不运行上游脚本。

## 输入、架构与训练配方

| 项目 | 上游证据 | 本地实现与裁决 |
|---|---|---|
| 输入长度/布局 | `U/ClosedWorld_DF_NoDef.py:37` 为 5000，`:41`/`:59` 为 channels-last 单通道 | `R/runner.py:35` 将已审核方向转 `[B,1,5000]`。`R/PLAN.md:9` 明确有符号时间戳取 sign，绝不当包大小；与方向型 DF 输入适配 |
| 类别 | 原 95 类，`U/ClosedWorld_DF_NoDef.py:40` | `R/df_reference.py:15` 为 102；这是本项目任务适配，原论文准确率不能直接比较 |
| 卷积 | `U/Model_NoDef.py:13`–`:30` 等：4 块、32/64/128/256、每块两次 kernel8 stride1 SAME、第一块 ELU、后续 ReLU | `R/df_reference.py:18`–`:21` 保留该配置；bias=True；stride1 偶数核 SAME 使用左3右4 padding |
| 池化 | `U/Model_NoDef.py:29`、`:44`、`:58`、`:72` 为 kernel8 stride4 SAME | `R/df_reference.py:9`–`:12` 显式 ceil 长度及非对称 padding，以负无穷排除人工边界；长度为 5000→1250→313→79→20 |
| 分类头 | `U/Model_NoDef.py:76`–`:90`：Flatten，512/512，BN+ReLU，dropout .7/.5，softmax | `R/df_reference.py:23`、`:27`–`:30`：20×256，transpose 后按 channels-last 展平，输出 logits。`R/runner.py:76` 用 CE，目标与 softmax+交叉熵一致；不声称有限精度完全相同 |
| 卷积 dropout | 每块 .1，上游 `:31`/`:46`/`:60`/`:74` | 本地 `R/df_reference.py:21` 相同 |
| 优化器/预算 | `U/ClosedWorld_DF_NoDef.py:33`–`:38`：Adamax .002，betas .9/.999，epsilon 1e-8，无 decay，batch128，30 epoch | `R/runner.py:55`–`:61`：torch Adamax .002、eps1e-8、默认 betas .9/.999、weight_decay0；每 epoch 重新 shuffle，保留尾 batch68。15300 条×30=459000 样本呈现，120×30=3600 updates |
| 原生选模 | `U/ClosedWorld_DF_NoDef.py:82`–`:88`：每轮 valid，直接评价末轮模型，未设 best checkpoint callback | 本地 `R/runner.py:82`–`:88`：20 次验证按最高 Macro-F1 严格提升保存，因此平局保留最早；同时保存末轮 `:92`。这是声明过的协议适配，不是原选模 |
| 数据划分 | `U/utility.py:14`–`:30` 直接加载预制 train/valid/test pkl；本脚本没有给出其采样预算或具体分割算法 | 不运行 loader，不导入原训练/测试集；沿用本项目 source150/valid510 的固定 manifest。不能假设论文训练量与150条/类相同 |

## 框架差异：哪些已修正，哪些仍未闭合

1. **BN**：上游以默认 `BatchNormalization`（如 `U/Model_NoDef.py:22`、`:78`），未在快照里钉住 Keras/TF 运行版本。本地 `R/df_reference.py:20`、`:23` 用 eps=.001、torch momentum=.01，对应常见 Keras momentum=.99 的旧统计保留系数，gamma/beta 的默认起点也一致。不能把这称为完整 parity：PyTorch train 方差用于归一化与 running variance 更新的估计口径不同；Keras/TF 的历史 backend/fused 路径需在确定版本后验证。eval/training 两种模式、BN buffer 更新都应做合成输入逐层对照，现有重载预测核验只证明本地可复现。
2. **初始化**：上游卷积默认 Glorot；三个 Dense 明确 `glorot_uniform(seed=0)`（`U/Model_NoDef.py:77`、`:83`、`:89`）。本地对所有 Conv/Linear 独立 Xavier uniform、bias0（`R/df_reference.py:24`–`:26`），随训练 seed 的 PyTorch RNG 初始化。分布族对齐，不复刻 Dense seed0 的随机流和跨 seed 相同初始化语义，也不复刻 TF RNG。
3. **Adamax epsilon**：数值参数 1e-8 **已经与上游显式参数一致**，不应误报上游 optimizer 默认 eps=1e-7 而建议盲改。注意它与 BN 的 .001 是两个不同常数。仍存在公式实现风险：已读取本地第三方 `.../site-packages/torch/optim/adamax.py:196`、`:277`–`:287`，PyTorch 使用 `u=max(beta2*u, abs(g)+eps)`。旧 Keras 常见写法为 `u=max(beta2*u,abs(g))` 后分母 `u+eps`，两者并非所有梯度序列上相同。本次未取得确切旧 Keras optimizer 源码/版本，故 Keras 一侧是待核实项，不能称已完成双向公式比对或把差异归为准确率瓶颈。
4. **padding/flatten**：现有 DFReference 明确修复并对齐结构；历史预检验证池化数值及长度，见 `R/PLAN.md:33`。尚无原 TF 和本地同权重前向逐层 parity 证据，因此称结构移植而非精确复现。展平顺序对随机独立训练的函数空间并非新增信息，但关系到同权重迁移/比较，不能随意忽略。
5. **损失、随机与设备**：logits CE 的数学目标一致，但不同 backend 的 softmax clipping、随机 shuffle/dropout、浮点内核不逐位相同。本地确定性及 FP32/no TF32 规则见 `R/runner.py:110`、`R/PLAN.md:31`。不应把跨框架同整数 seed 视为同初始化或同训练轨迹。

## 与更早本地 DF 的区别

`src/ta_wf_next/models/df.py:7`、`:10`、`:13` 使用对称 padding=4，kernel8 时每次卷积增加1个位置；`:16` 池化无 padding。四阶段长度相应为1249→311→77→18，不是官方20；`:34`/`:47` 分类头固定18×256。卷积和前两个 Linear 无 bias，BN 用 torch 默认 eps/momentum，默认初始化也不是显式 Glorot，`:46` 直接 channels-first flatten。`src/ta_wf_next/models/SOURCES.md:10` 已明确旧来源不能称作者官方实现。

因此：较早约48%的 DF 及其融合阴性结果属于旧实现/旧预算历史，不能与新 DFReference 混为一个基线，也不能为了“重新复现官方DF”重复此前已经完成的 padding/bias/flatten 修正。

## 可直接复用的结果

`R/RESULTS.md:8`–`:13`、各 `R/artifacts/df_<seed>/report.json`：

| seed | F1选中步数 | source accuracy | valid accuracy | valid Macro-F1 |
|---|---:|---:|---:|---:|
|21729|3600|90.588%|71.176%|70.156%|
|23407|3600|90.248%|71.569%|70.190%|
|22026|2880|85.039%|69.608%|68.582%|
|平均|—|88.625%|70.784%|69.643%|

参数 3,982,502；报告保留 last valid accuracy 为71.176%、71.569%、68.431%（平均70.261%）。三任务完整重载及独立 sklearn 核验步骤见 `R/runner.py:95`–`:103`；本轮未重跑该核验，不把历史核验计为新实验。

本轮仅对已存的20点评估 `history.json` 做 JSON 级审计：三 seed 最高 accuracy 的最早步数恰好仍为3600/3600/2880，与当时 F1 选出的 checkpoint 相同。因此已有 DF 数字可用于新 accuracy 目标的历史参照，但必须声明此一致性是事后核对，不改写原选模计划。若后续方法使用新的 accuracy 选模规则，历史其他方法也须做同样资格核对，不能默认等价。

两个 DF seed 的最好值在30轮末端，说明本配置未证明充分收敛；也不能据此断言延长必能提高或达到90%。新 CNN 最后验证退化的证据不能替代 DF 自己的曲线证据。

## 后续有界公平比较建议（仅方案，本次未执行训练）

- 首选将此 DFReference 作为已有方向基线复用；把新增预算投入审计后合格的强模型，避免重复无变化 DF 三 seed。
- 核心条件固定 source150=15300，valid510，102 类，方向前5000及原 padding，训练 seed21729/23407/22026，同一抽样清单；source 标签 CE，valid 仅预定选模/评价，无未来日期、无外部数据、无 query 适应。无模型特定增强的官方配方比较与“含遮挡的当前最佳”应分清方法包，不能归因纯架构。
- 新主指标为三 seed mean valid accuracy；Macro-F1、逐seed、训练拟合、末轮指标、参数量、样本呈现量、updates、时间同步报告。三 seed 不是三份独立验证集；固定 valid 已反复开发，90%只是开发目标。
- 新候选统一20个预定 checkpoint 选择机会，最大 accuracy 平局最早；每个模型允许已审计的原生优化器/调度，在训练前冻结预算，明确不等FLOPs与样本呈现次数的限制。达到90%的判定使用三 seed 均值，不选最佳seed，不用集成冒充单模型平均。
- 若审计后认为 DF 收敛性值得单独检验，可另冻结一个 **60 epoch 从头训练**条件（同 batch128/Adamax，总7200 updates、918000样本呈现，20点评估可预定每3epoch一次），仅3个seed、无其他参数搜索、单任务45分钟/该支线总2小时硬上限；它是扩预算开发候选，不是重做已完成基线，也不能与30轮基线按等计算量称公平。其执行需要纳入根任务最终冻结的总预算，不因本报告自动启动。
- 原 Keras 精确复现优先级较低：若确有需要，先固定合法可用的框架版本，再用合成输入/固定权重检查 SAME、flatten、BN train/eval 和多步 optimizer 轨迹，合格后另立版本；不把普通数值差异预期为20个百分点收益。

最终推荐：**DF已具备可复用、声明清楚的移植参照；不要把精确Keras复刻或无界延长DF作为通往90%的主要假设。** 应与更强候选完成同权限协议审计后，形成有限的多模型比较计划，再训练。
