# 分类器与可学习生成器：共享对照的CPU并行实验

状态：frozen；用户已明确授权按计划执行。已完成实现、结构/恢复/输入检查及合成吞吐，正式训练即将启动。CPU方案，所有正式训练入口拒绝draft。

## 问题与既有证据

当前优先目标仅为提升基础分类器。生成器应在压缩之前根据source分类损失学习原始特征到token的转换。上轮固定2040/510、100轮、三seed实验summary/ordered valid accuracy26.471/18.497%、Macro-F1 24.655/17.162%，保序方向输入训练拟合更高却泛化更差。既有训练与核验已完成，不重复；作为明确标注的历史开发参照，不算本轮独立重复。前次100轮末ordered训练accuracy82.533%，故本轮优先区分局部编码、汇聚和生成器更新速度，而非无限延长轮数。

## 唯一实验目录与数据权限

四条件组成同一组共享对照实验，全部配置、日志、模型、预测、诊断归属本run；每条件×seed有唯一文件前缀。复用上一轮artifacts/prepared.pt（仅source/valid）与既有抽样清单，使用configs/datasets.json核对路径与输入来源，不导入旧项目代码、不载入旧模型权重。5000包方向、source2040标签、valid510标签、102类。不重新划分；source用于梯度和诊断，valid仅按预定规则选模与开发比较，未来日期和外部数据不读。

## 可学习生成器

原始方向+有效mask两通道，连续5000包上两层共享Conv1d（2→16→16，kernel5、stride1、padding2）与GELU，每层输出重新应用mask，防止padding经偏置传播。暂不加入BN或随机dropout以便比较汇聚/冻结。连续提取后每50包分成5个有序的10包子区间；每区间产生16维局部摘要，保留子区间顺序，共80维。拼接原均值/切换比例2维和5个有效比例，共87维，再由生成器自己的Linear(87,52)生成100个d52 packet token。全空子区间输出严格为0，attention softmax仅覆盖有效位置，严禁全padding产生NaN。

后续沿用本项目层次化Transformer（d52/4heads/FF104/dropout0.1），packet入口投影改Identity；run128和window50/250的输入与处理不变。生成器G的参数边界包括两层卷积、汇聚score和87→52投影；不得把投影留在分类器却声称整个生成器已冻结。所有条件实例化同一参数结构；mean条件禁用并冻结score。共享Linear(16,1,bias=False) score初始化为0，因此初始attention为masked mean，四条件同seed在第一步前token/logits应完全一致。总参数相同不等于可训练参数/有效容量相同，须分别报告。

## 四个独立可并行条件

| 条件 | 局部编码/投影 | 子区间汇聚 | 生成器学习率 | 主要用途 |
|---|---|---|---:|---|
| A local_mean_joint | 联合训练 | 固定masked mean | 0.001 | 可学习局部编码的简洁基线 |
| B local_attention_joint | 联合训练 | 可学习masked attention | 0.001 | 分类反馈同时调整特征和token汇聚 |
| C local_attention_frozen | 初始化后全冻结 | 初始attention全冻结 | 0 | 检验反馈学习的增量 |
| D local_attention_slow | 联合训练 | 可学习masked attention | 0.0003 | 检验慢速生成器更新是否改善泛化 |

所有条件分类器lr0.001、AdamW、weight_decay0.0001，无scheduler。A/B生成器weight_decay0.0001；D为0.0001/0.3，使每step乘性衰减lr×weight_decay与B相同，主要操纵梯度更新尺度；冻结C无优化器组。score禁用/冻结按显式参数名配置。不同条件采用同seed同初值（先建统一state再切开关）、同batch顺序和相同标签权限。

共享对照：B−C回答生成器是否从反馈中受益；B−A回答可学习汇聚的增量；D−B回答慢更新是否改善泛化。C是随机初始化生成器冻结的机制控制，本身偏弱，胜过C不能证明方法实用或独特。A与历史直接保序差别含编码结构/容量，不能唯一归因为共享局部模式；历史摘要作为实际性能参照，不能把它算成新重复。

## 训练与选模

seeds1729/3407/2026，各100epochs、batch64；每5epochs在eval模式分别计算全部source/valid accuracy和macro-F1，每seed选最早最大valid F1的checkpoint。每条件3seed，共12组；不会根据第一个seed表现取消其余seed或加开变体。优化目标仅source CE，valid/未来不能提供梯度。主指标macro-F1，同时报告accuracy、训练/验证差、预测类别数、每epoch曲线及耗时。

每项机制比较预定筛查门槛：平均valid F1至少+1.0pp、3/3seed为正、平均valid accuracy不下降。三项均报告，不能挑有利的比较；它们是开发筛查，非显著性检验或独立确认。实际采用还须看是否超过历史摘要26.471% accuracy/24.655% F1及seed一致性。未预设“接近DF”的比例线，也不将胜过弱冻结控制等同达到成熟模型水平。没有统一实际收益时保留摘要方案，不自动启动后续搜索。

## 并行与资源预算

计划六个CPU worker、每worker2个线程（interop1），合计12训练线程；12组按seed轮转排队，约两批任务量。独立结果核验最多使用释放出来的2个CPU线程，始终不超过12线程总额。准备阶段共享只读输入，避免六个worker重复解压原数据。不得启动GPU进程，不抢占/结束其他人的任务。

每seed累计3600秒，epoch边界停止（最多超一个epoch），监督进程保留180秒收尾；总pipeline墙钟上限7800秒，正式训练预算最多24 CPU线程小时，任务总RAM建议上限24GiB、单worker4GiB。预算耗尽时保留不完整结果，不追加seed/epoch。CPU局部编码成本尚未实测，上一轮约17分钟不能当作本轮耗时承诺。正式运行前用合成数据测全部条件及6worker聚合吞吐；如竞争明显，仅在冻结前降低并发，不根据真实valid结果改变预算。实际线程、并发、测量依据及停止规则记录并冻结后才开始训练。

## 与训练并行的检查

固定source小清单（每类首样本中取前32类，不增加标签权限）在eval模式测生成器token的初始/当前变化、生成器梯度范数和参数变化；该清单只作机制诊断，不按它选择模型。C生成器参数/token必须保持不变；A/B/D应有有限梯度并确实更新，B/D汇聚score也须能得到非零梯度。生成器学习的必要条件通过不等于分类有效。每轮保存latest（含模型、optimizer、RNG、累计耗时），valid改善保存best；每个结束任务可即时CPU重载复算source/valid预测及独立指标，最终应为24/24组。

预执行测试：同seed四条件初始token/logits一致；padding/partial/batch独立；全空mask有限；冻结边界与梯度正确；checkpoint恢复一致；数据散列和source/valid重复检查。失败只允许记录修错，不借机修改候选目标。所有seed完成或停止后统一报告结果与失败，更新EXPERIMENTS.csv、RESULTS.md、STATUS.md。

## 依赖与当前进度

上述四条件均不依赖另一条件的训练结果，可同时开始；验证脚本依赖各自checkpoint完成。共同前置步骤已于配置冻结前完成，见下文；运行状态由STATUS与queue_progress记录。

## 执行前核验与资源冻结（2026-09-26）

生成器位于src/ta_wf_next/learnable_generator.py。本项目先前CPU表示实验的恢复/登记逻辑被显式改写为本run独立common/train/pipeline；不导入先前run作为训练模块，更无旧项目代码依赖。运行环境显式使用/home/rbf/TA-WF/.venv/bin/python的PyTorch，不使用其模型或checkpoint。

四条件总参数119578，生成器总参数6064。A可训练119562（生成器6048，score16参数禁用）；B/D可训练119578（生成器6064）；C可训练113514（生成器0）。总容量相同但活动参数不同是有意消融，不能混称训练容量完全相同。

单worker2线程合成训练步中位A/B/C/D为0.382/0.406/0.316/0.415秒；6worker同时约0.364–0.440秒，聚合13.375步/秒。仅合成前反向，没有optimizer更新/真实分类评分。保留6worker×2线程、每seed3600秒、pipeline7800秒上限。估计正式两批约45–70分钟，非完成承诺；不据实际valid追调资源。训练/即时核验共享6个slot，核验不会额外占用训练上限之外的线程。单worker RSS4GiB/总RSS24GiB由监督进程检查；verify每任务另有300秒时限，仍受全程预算约束。

新增4项生成器测试通过；全项目unittest47项运行46通过1项既有parity跳过。覆盖四组同seed初值/token/logits逐位一致、局部padding/全空mask有限及batch独立、CE梯度传至conv/score/projection、完整冻结与mean条件score隔离。额外合成恢复检查验证AdamW+dropout+三种RNG恢复后参数逐位一致。输入复用散列、原始路径映射、行索引、标签计数及patch摘要重算通过，source/valid方向交叉为0；本次不再打开原始数据或未来文件。

pipeline.py按seed轮转启动train.py；每任务结束优先占用释放的slot执行verify_one.py，再继续队列；finalize.py统一审计24组source/valid预测及所有比较，写RESULTS并更新EXPERIMENTS/STATUS。检查与合成产物不作为真实效果。配置和运行依赖散列记录在artifacts/freeze.json。
