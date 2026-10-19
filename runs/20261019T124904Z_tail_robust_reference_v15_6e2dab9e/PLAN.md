# 低风险目标×扰动稳健×reference重加权 v15

用户授权A/B/C全部尝试。三seed42/100/2026，已观察version_045/network_USA两片；固定v9 smooth_balance候选、v12基线，v14作为历史对照，不重新训练骨干/LoRA/辅助编码器。不是独立确认。

源输入直接复用v14固定2448 train、816 valid及五种扰动、三维X=(margin,candidate_logprob,third_margin)。所有输入文件和代码/config在拟合前SHA256冻结。源原始trace跨角色隔离，每个trace的五种扰动不是独立样本。源骨干已见源train，非模型外预测；本轮不能消除此局限。强扰动保标签是假设。

八组全因子：ce(原目标)、A(局部尾部排序)、B(组稳健CE)、AB、C(reference加权CE)、AC、BC、ABC。指定ABC为组合主臂，A/B/C为分别预注册检验，不用最好一组替换主臂。所有组相同三维源标准化和固定.25归一化残差融合；另报分支单独排序以定位融合损失。ce须数值复现v14线性模型与评分到1e-4，无超参搜索。

CE：三类balanced交叉熵，L2=1/N_changed，bias不惩罚。B：按case_id计算组损失，temperature=.5的平滑最大值，组先验为各组样本权重质量。A：每组R与v12最高分D构成pair，D数量ceil(.01*该组全部frozen正确数)，并非changed负例的1%；R/W另以.25权重构成pair。每组每类pair固定散列/确定性种子至多2048，按exp(-|v12_R-v12_neg|)局部加权，优化softplus(h_neg-h_R)，L2=.01。A的训练目标是分支局部排序代理，不声称直接精确优化目标pAUC。B对pair组损失作相同平滑最大值。最终所有分支u均用源changed train均值/SD标准化，再以v12+.25*z(u)融合；不扫描融合权重。优化固定scipy L-BFGS-B最终解，maxiter1000、ftol1e-10、gtol1e-7，无目标选择。

C：仅从既有1024独立reference缓存重建同三维描述，使用v9冻结与适配logits、v11三seed五视图辅助logits和v12固定温度权重；不访问reference网站标签。仅changed样本用于域分类，balanced logistic C1，5折GroupKFold，源按原trace分组、reference按原行分组，产生源OOF密度比p(domain=ref)/(1-p)。在源与reference平衡先验下解释为边缘特征密度比，非误改概率。raw及[.1,10]剪裁权重和ESS全部报告，训练权重归一均值1。至少32 changed reference，raw ESS/N>=.1、clipped ESS/N>=.2、reference域概率<.95比例>=.5，否则C相关状态记不可行并保留失败，不以基线冒充执行。C仅重加权监督损失，sourcevalid阈值沿用所有臂统一未加权规则。共变量漂移假设不保证成立。

逐扰动留出：仅四个不含C组，train的其余四种扰动训练，在valid的被留出扰动评分；共120微模型，无query选模。原始trace train/valid隔离，留出扰动并不构成新数据确认。不据留出结果挑选目标算法。

源valid阈值统一最大U=R-D-.25W，NAR<=1%、R>=10、U>0，否则拒绝全部。源阈值和全部48最终模型、120留出模型的参数/归一化在任何本轮query评分前冻结。query输入与标签不参与统计拟合；只复用v13query特征在终端计算指标。两片采用既有内容支持数值身份假设，USA仅地域诊断。WTT/AWF关闭。

主指标：pAUC(FPR<=.01)/.01，FPR分母全部frozen正确者，未改写正确样本给低分哨兵；另报确定性Recovery风险面积、Recovery@NAR1、同v12实际D预算Recovery、同v12纠错R的D/W与可行seed。诊断阈值不部署。报告全部11评分×6状态与源阈值真实NAR/Recovery/Accuracy/MacroF1。低风险同分组不可拆，匹配不足3seed均值NA。

每组通过需两片pAUC均值不下降、Recovery@NAR1下降不超.1pp；且各片同实际D预算Recovery>=+.2pp或同R全部seed可行、D不升且W<=.95基线；各片至少2/3seed同预算Recovery不下降。A/B/C各自预注册检验；ABC是组合主判据，无事后替换。完整矩阵和阴性保留，不扩片。另源阈值最大NAR<=1%且每状态R>0作为观察性部署资格，非安全保证。

预算CPU2线程、wall1800秒、168重排模型(48最终+120留出)、36域模型(6×(5折+全量))、输出1GiB、零GPU训练/新前向。源/目标模型重载全分数与指标复算；所有输入/拟合对象散列复核。预算/非有限/隔离/重载失败停止，错误修复记录原因，不按结果增补方法。最终RESULTS与STATUS登记，不部署。

源预结构检查：clean组没有R，多数无changed。pair组仅在有R且有负例时贡献损失；被留出clean的pAUC若无R则保留undefined，不冒充零。该处理在首次拟合和目标评分前声明。

实施修复1：首次源预检将不同train/valid文件内row_id作全局比较而误报；尚未拟合。恢复为prepared timestamps内容SHA256隔离，并以内容组做域分类OOF；新增读取仅原获准源缓存，未改样本、损失、超参、预算或判据。失败日志与旧freeze保留，query未评分。
