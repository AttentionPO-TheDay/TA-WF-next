# 20260922T022343Z_cpu_run_order_ablation_5c3b1b73

问题：相同容量CNN下原序与固定乱序run的差异能否跨三个训练seed复现

状态：frozen，数据访问前冻结。

同容量同初始化同minibatch顺序下比较原序与固定乱序。exact/coarse各3训练seed(1729,3407,2026)×2条件，共12次从头训练。历史1729原序仅作复现检查，不是新的独立seed。

显式复制修改本项目cpu_ordered_run_encoder的train.py及verify.py。相同两层k5/64通道CNN和128隐藏层head、mask、source逐通道标准化；15epochs/batch128/AdamW lr0.001 wd0.0001。CPU4线程，GPU隐藏，timeout1200秒。非有限、隔离失败或超限停止，保留失败记录。source valid macro-F1逐轮选模，平局取早；完整报告全部seed。

沿用固定样本清单source2040/valid510/JP2036/subpage2040，前5000观测、512run槽。source监督，valid选模，目标标签仅全部预测封存后评分。configs/datasets.json定位共享只读数据。无新条件、WTT/AWF或旧checkpoint。分层抽样是离线设计，不是部署策略。

截断和标准化后，逐行排列有效token的完整向量，padding不动。角色source/valid/jp/subpage分别用numpy RNG 9001+角色序号，按冻结样本行次序生成排列；exact/coarse共享，全部训练seed共享。训练和评价都乱序，不是只在测试破坏输入。保存permutations.npz。mask、token多重集和标准化相同，配对模型初始化及batch次序相同。

coarse的邻居bin差和边界标记随token移动，仍保留局部顺序线索；只消除token排列，不声称删除全部顺序信息。exact边界标记也保留。512截断末token的next-bin使用观察预算内第513run沿用旧规则。固定一个排列seed，不代表对所有乱序重复稳健。训练seed不是独立日期或数据重复。

主分析：各角色各视角ordered-shuffled accuracy每seed配对差及均值/样本标准差；完整报告macro-F1。预定仅当某视角valid与JP各三个seed正差时称跨seed一致，否则混合或无一致收益。subpage单列。不能据此声称优于packet/window或抗时间漂移；JP/subpage是网络/行为条件。看到目标结果后不续训或调参。
