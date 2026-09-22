# 实验结果

## 本轮结论

完成首轮结构/provenance 审计，不是训练就绪认证。44 个原始 NPZ 的字段、形状、dtype、归档成员 CRC 和文件大小均与旧审计一致，0 metadata mismatch。没有加载数组值、标签数组、模型，没有训练/评分，WTT-Time/AWF 未打开。ZIP/header 读取可能触发底层缓冲预读，不声称数据字节物理上完全未被访问。

新证据见 artifacts/header_audit.json，脚本 audit_headers.py。旧标签计数和内容匹配继承自以下记录，并未重新计算：
/home/rbf/TA-WF/outputs/proteus_six_dataset_protocol_audit_exp_d839f7ce217f488a/proteus_six_dataset_protocol_audit.md
对应 evidence/local_npz_audit.json 的 SHA-256 已写入新产物。归档 CRC 一致不等于重新验证 payload 完整性。

## 场景可用性

| 场景 | 当前证据 | 可以研究 | 限制与下一门槛 |
|---|---|---|---|
| Temporal | Day0/14/30/90/150/270，102 类，139667 行（旧计数） | 跨日期综合变化 | 多轮开发已暴露；不能证明变化由网站内容更新单独引起 |
| Version | Tor 0.4.5/.6/.7/.8；每个目录 train/valid/drift 合计100162行 | 指定 source view 对聚合 drift 的鲁棒性；Tor 版本在用户目标内 | 不是浏览器版本；drift 无逐行 version；四视图不可当400648独立样本。明确 A→B 需补 provenance 或验证恢复 |
| Network | SG source；SG/JP/USA/DE/UK 文件，各102类，合计34700行 | 采集地点/网络条件变化 | 地点不是独立操纵的 RTT/带宽/丢包；需内容去重及条件内划分 |
| Behavior | homepage→subpage；subpage有URL，20400行、17739独立URL（旧证据） | 页面访问行为/页面变化 | 不是网站版本更新；同站有重复URL，需URL层面的隔离审计 |
| OpenWorld | pooled background 102 + 102 monitored | 未知类/背景混入下的鲁棒性 | 不是独立漂移原因；无背景站点身份；未来 monitored 重用 Temporal |
| Defense | 7个时间文件；形状/计数与Temporal对应 | 暂仅推定WTF-PAD下的时间变化 | 防御身份为旧证据强推断，无逐行防御字段；本地缺obfs4/Front |

Network/Behavior 的 train、valid、SG/homepage-test 在旧审计中已证明字节相同。因此它们提供不同目标条件，但不能作为独立源采集重复。全六组数值类别对应仅有部分直接证明，不跨组直接共用分类头。

## 输入与标签权限

44个文件均为 X(N,10000) float64、y(N,) float64；只有 Behavior/subpage 多 url 字段。signed relative timestamps 的语义来自既有官方处理代码和旧抽样数值证据，本次 header 不能独立证明语义。无原始报文字节、packet-size channel、session/capture ID、TLS边界或网页资源身份。符号的客户端上下行物理定义未闭合，不硬编码 outgoing/incoming。

Proteus 全组可共用方向+时间接口，不必为Tor/网络/行为分别写 tokenizer。跨AWF/WTT扩展时共同接口仅方向；时间/大小必须是独立可选属性，不能混为同一种长度。现有有效长度/首个0/padding、时间单调性、singleton duration=0、窗口截断需在正式 token 生成前审计，不能按第一个0盲目终止。

旧审计读取过六组标签、抽样X及部分内容匹配，所以不能称它们完全未见。Temporal性能暴露明确；其他组是否已被旧训练/选模使用尚未穷尽检索，不授予独立确认资格。元数据审计本身不等于性能调参。

## 生成器：共享结构优先，泛化尚待证明

建议区分三层：输入语义适配器→共享结构tokenizer→可学习编码器/适应机制。适配器处理字段差异，而不是按漂移原因选择算法。tokenizer采用同一观测预算、边界定义、方向run和顺序规则；可选输出粗粒度run长度、邻域规模关系、时间属性及缺失/删失标记。此为候选，不是已冻结方法或创新结论。

Tor版本可能改变调度/打包，Network可能影响时序，Behavior可能改变资源组成；这些是可检验假设，不由数据组名证明。相对规模可减弱共同缩放却也丢掉类别信号；局部插入删除会破坏位置；单向run方向交替，方向掩码任务容易退化。共同 tokenizer 不等于表示天然稳定，升维/RLE不创造信息，精确长度旁路也不能保证粗化不变性。

不建议初始一因一生成器：部署通常不知漂移原因，混合变化难路由，易变成多套独立调参。若确需条件分支，应以可观测输入统计驱动，冻结路由且不使用真实类别或人工原因标签；条件已知的专属模型只作显式oracle/专属基线，不冒充通用方法。

未来泛化证据应包含：共享方法在每场景自己的source训练（证明流程可复用）与未参与设计/调参的条件留出（更强的跨条件泛化）。每场景独立选好参数再汇总不证明统一机制。暂不开放具体测试条件或训练任务；先审计既往暴露、冻结开发/留出清单。若采用积累无标签同池适应/识别，需另立transductive协议，不继承默认独立query权限。

## 待完成的训练前门槛

1. Version：尝试用四个目录中有版本身份的 train/valid 作锚点，与 drift 做完整trace精确匹配。须区分唯一匹配、跨版本冲突和未匹配；不能用分类器猜版本充当真元数据。恢复部分不代表恢复全体；使用目标版本视图的train作为评价候选时，须排除所有训练重叠并重新冻结角色。更可靠替代是获取官方per-trace version manifest。
2. Network/Behavior：完成内容级重复/冲突、URL重复与source/target隔离审计，核对时间/长度数值质量及既往性能暴露，再建明确清单。元数据缺失时不能证明session独立。
3. 网站版本/浏览器版本：当前无受控前后版本证据，不用日期或subpage代替。用户已接受Tor版本，因此不阻碍研究范围。

本轮停止于结构审计与设计建议；未实现生成器、未训练、未建立性能对照或选取稳定区域。
