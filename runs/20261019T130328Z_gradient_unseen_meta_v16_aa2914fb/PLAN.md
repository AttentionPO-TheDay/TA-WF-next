# 梯度未见纠错监督检验 v16

用户授权继续检验“骨干未训练过的样本能否提供更可靠纠错监督”。本轮为有界低成本必要检验，不训练新骨干，不宣称完成严格交叉拟合。三source seed42/100/2026、已观察Version045/USA、固定v9 smooth_balance候选与v12评分接口，辅助网络和源BN全部冻结。

角色：源18553 train与官方2160 valid均已观察。valid虽未参与骨干/辅助编码器梯度，却参与过历史checkpoint与温度/权重校准；因此仅有gradient-unseen资格，非完全模型外/独立确认。将valid用于小重排器监督为本轮显式角色变更，不能再将这些角色当作未参与方法训练的验证。

按prepared.pt有效TAM内容SHA256全局分组，冲突标签停止。每类唯一group按sha256(v16|valid|group)排序，前6供GU元训练，接5供阈值校准，接6供源终端诊断：612/510/612行。任何group不跨角色；不足17停止，不重抽。IM元训练从v14固定2448 train行的唯一TAM组按sha256(v16|train|group)每类取6(612)，与三valid角色内容隔离。不同文件内row_id不是全局身份。所有清单在任何新前向/拟合前冻结；非随机重划，不授予新确认资格。

五臂相同三维X=(v9 margin gain,candidate mean logprob,candidate-vs-third margin)，相同class-balanced multinomial LogisticRegression C1/L2、lbfgs maxiter2000 tol1e-8，固定最终解。IM_all源训练clean+四扰动，GU_all梯度未见meta clean+四扰动(主臂)，IM_aug与GU_aug各移除clean作成对消融，GU_clean只用自然clean错误作低样本可行性诊断。主要四臂每类R/D/W至少10，GU_clean至少3，不足记该状态不可行；主臂不足则主判据失败，不增样/改阈值。每臂元监督原始标签预算均612行，五扰动不是五倍独立样本；clean-only样本机会不同，不能作为主要公平对照。

扰动严格复用v14五种定义及row盐：train用既有缓存，valid四种合成用v14的valid salt97，保标签是假设。valid clean通过原v9 main/aux三seed五视图缓存取出以消除新批次差异；完整2160 valid的main/aux原始输出先逐位复现。源train四臂仅重用已审计v14缓存，不新前向。新valid合成15缓存在GPU0/1/2各一seed生成，缓存每个前256行独立重算逐位一致，模型参数/缓冲前后逐位不变。

证据温度/权重与v12源尺度保持冻结，共同支持probability/log-odds定义；sourcev14重排参数仅历史对照。每个新模型以自身changed meta样本标准化三维特征与u=gR-logsumexp(gD,gW)，最终v12+.25*z_meta(u)。不扫描融合系数。校准仅510 observed source-valid calibration行的五变体，统一最大R-D-.25W且NAR<=1%、R>=10、U>0，否则拒绝全部。所有模型/尺度/阈值在打开本轮query特征评分前散列冻结。源诊断612行只终端评价，不选模。源valid全部历史曝光不撤销。

query仅v13已封存逐行特征，标签只终端指标，不进行任何归一化/CDF/风险模型拟合。WTT/AWF关闭。reference只继承v9旧适配，无新目标信息拟合。Version/USA沿用既有内容支持102类对应假设，非官方身份；USA仅地域诊断。

指标：低NAR pAUC/.01、确定性Recovery面积、Recovery@NAR1、同v12实际D预算与同IM_all实际D预算纠错率、同v12纠错数的D/W及可行性。NAR分母全部frozen正确者，非changed负例；W单列，同分不拆。source-calibrated target实际NAR/Recovery/F1/Accuracy另表，query最优阈值仅诊断，不部署。全部臂/seeds/阴性保留，缺seed均值NA。

机理门槛：GU_all相对等预算IM_all在两个切片的同IM实际D预算Recovery均值>=+.2pp，pAUC均值不下降，各片至少2/3seed同预算不下降。方法升级另需对v12达到旧v14主门槛(两片pAUC不降/Rec1不低于-.1pp，并同实际D预算Recovery>=+.2pp，或同R全seed可行且D不升W<=.95基线)。仅GU比弱控制好不代表超越v12。不按目标反馈挑GU_aug替换主臂，不扩片。

预算：wall1800秒、三GPU各worker<=600秒，总保守GPUh<=1、2CPU线程/进程、<=30微重排器、15新valid缓存、1GiB。没有骨干/LoRA/辅助神经网络新训练。前向预检、缓存首256重提取、30模型全部source诊断/query分数重载、源阈值及目标计数独立复核、源角色内容隔离、冻结hash核验。预算/非有限/隔离/数值复现失败停止，修错留痕，不按效果改方案。
