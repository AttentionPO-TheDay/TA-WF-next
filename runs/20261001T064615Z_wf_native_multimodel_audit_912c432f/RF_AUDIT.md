# RF作者实现与Proteus配方审计

推荐：作为方向＋时间聚合的原生方法候选，不能混入纯方向架构公平比较。不是随机森林；RF指Robust Fingerprinting。先完成TemporalDrift source/valid时间与TAM接口审计，再冻结训练。

## 固定来源

作者仓库 https://github.com/robust-fingerprinting/RF ，README明确对应USENIX Security 2023《Subverting Website Fingerprinting Defenses with Robust Traffic Representation》，作者Meng Shen/Kexin Ji等。固定commit `b75680148429c5f554114b78a0f846541b39f0ec`；公开源代码文本保存artifacts/rf，provenance.json记录URL及SHA256。未下载数据、预训练权重或执行上游脚本。

Proteus发布代码固定commit `4cdab4163bf3de7036a2498dac533c888b97664d`，本次明确复用此前审计的归档/tmp/proteus_audit.HOKVvy源文本并保存必要文件及来源到artifacts/proteus_rf；不导入旧训练器、不修改共享数据。作者RF与Proteus对RF的集成配方必须区分。

## 输入及模型

- 作者RF/const_rf.py定义5000包上限、TAM1800列、最大80秒、95类（本项目须改102）。RF/FeatureExtraction/packets_per_slot.py:3–19对每包按方向分别累积，bin=int(time*1799/80)，>=80秒放末列。不能把列数1800解释为1800包，或把有符号时间戳当包大小。
- TAM形状2×1800，送模型时B×1×2×1800。它需要每包时间；direction-only缓存不足以生成原生TAM。数据权限须明确增加源期raw时间输入，仍限同一前5000包及同样训练/valid行。
- 作者RF/models/RF.py：四层二维卷积后reshape为32通道的一维序列，后续128/128、256/256、512、num_classes卷积与BN/ReLU，最终AdaptiveAvgPool1d(1)。最后类别卷积也保留BN/ReLU，不能无声改成普通linear logits头；初始化只显式覆盖Conv2d、BN2d及Linear，Conv1d保留框架默认，不要“统一修正”。Proteus模型结构基本保持该流程，主要添加features返回接口；此处为静态代码对照，未做数值parity。
- 全局交错时间回退不自动阻止按每包自身时间落bin，因为TAM不依赖相邻时间差。仍需核实TemporalDrift本身的非有限值、padding、time单位、负/零、80秒截断与计数守恒，不排序后重新截前5000包。
- 已有20260921T190311Z时间审计只覆盖Version/Network/Behavior，明确未读TemporalDrift，不能把其方向内单调结论直接套到本轮。
- 既有tam_day90转换不一致，不读、不复用任何未来TAM；source/valid也应在本run用明确规则由raw重建并核验，不因文件名存在而直接信任。

## 配方差异是实质问题

作者RF/train.py：30epochs、batch200、Adam lr5e-4 **weight_decay=.001**；每轮按 `lr=5e-4 * 0.2**(epoch/30)`衰减（epoch从0开始），CE，最后保存，官方脚本不使用本项目式best-valid选模。RF README原DF数据为95类×1000条，本项目仅102类×150条，论文成绩不是本项目保证。

Proteus scripts/TemporalDrift/RF.sh：30epochs、batch200、Adam lr5e-4、TAM1800、valid F1选模。其exp/train.py:95仅传lr，**没有作者的weight_decay=.001**，脚本也未提供作者上述指数LR计划。故不能把两者都写为完全相同“原生RF配方”。本轮优先作者配方作为候选，Proteus版本如以后运行应另标版本，不能根据结果择优包装。

作者train_10fold.py和README的10折/随机9:1清单不用于当前已观察数据重划分。作者脚本val函数的分batch文件覆盖/打印不是可靠全局评价入口，应使用本项目独立预测汇总；不得直接运行带其他数据路径的上游脚本。

## 有界后续方案（draft）

1. source150固定15300条、valid510固定行；仅读取train.npz和valid.npz对应原始时间及标签权限，source训练/valid选择。TAM input adapter先过合成边界0/80/大于80、双方向、padding、计数守恒和逐行来源对齐检查。
2. 作者网络102类从头初始化，seed21729/23407/22026，30epochs batch200 Adam5e-4 wd.001及作者指数衰减，不额外调bin宽度/80秒窗口。
3. 作为固定划分适配，20个预定epoch验证（1..10,12,14,16,18,20,22,24,26,28,30），按accuracy取最早best，同时报告last及Macro-F1；这是本项目选择规则，不是作者最后一轮规则。
4. GPU0单RF任务，上限1小时/seed、3GPU小时；确认显存后可与轻量任务最多2并发，其他GPU不使用。禁止未来评分、不做Proteus适应。
5. 它回答“原生时间聚合方法在本项目能达到多少”，不回答“仅换网络是否提高”。若需要机制归因，还需同输入权限对照，不能把RF对方向CNN的增益当纯网络优势。

根目录tree未发现LICENSE文本，仅README research purposes说明，复用时保留作者归属，不擅自声明开源许可证。本轮只审计未分发或执行。依赖清单有旧版torch/numpy，运行前应做兼容性核验，避免无关countermeasure依赖进入最小模型。
