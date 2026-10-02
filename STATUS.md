# 当前进度
当前wide+BN窄优化配方对照：`20261002T062832Z_wide_bn_recipe_refine_37f6ab67`。9项新训练+3项历史基线完成；baseline89.150/lr05 88.301/wd05 88.954/组合87.582%，三种新配方均FAIL。汇总脚本误用上轮比较键，已从完整产物恢复汇总；47hash、48组指标、初始化/索引/20次选模复核通过。保留89.150%候选，停止本轮配方扫描；冻结生成器权重跨头复用尚未启动，未来关闭。
当前逐级生成器容量BN与Mixup对照：`20261002T053706Z_progressive_capacity_bn_mixup_1ad233da`。12项新训练+3项历史基线完成；baseline=86.536%/wide=85.882%/bn=88.366%/wide_bn=89.150%/mixup=87.582%；通过候选['bn', 'wide_bn']；60组预测指标和冻结hash通过。
当前网站指纹生成器跨头对照：`20261002T045631Z_wf_generator_head_transfer_4ed54f62`。12新+3历史完成；旧Transformer77.647/旧MLP71.373/新Transformer86.536/新MLP84.641/RF同配方91.699%。新生成器在两头均3/3提升（+8.889/+13.268pp），跨头结构门槛PASS；Transformer对新MLP+1.895pp且3/3正。自有开发基线升为86.536（距90%3.464pp），RF91.699单列强对照。48hash/60指标组完成后复核通过；下一候选冻结生成器权重跨头复用尚未启动，未来关闭。
当前下一轮设计复审：已复核原计数完整保留、旧加深为1×1残差的限制。建议逐级跨时间卷积生成器×Transformer/逐tokenMLP后均值2×2，另加同log1p/AdamW/mask配方RF对照；12新任务+明确历史参考，现已完成并复核，结果见顶部生成器跨头run。详见最新soft_targets_mixup RESULTS.md设计复审；未来关闭。
当前软目标与Mixup对照：`20261002T032948Z_soft_targets_mixup_3e1aa98f`。12新+6历史完成；CE77.647/打乱78.824/均匀77.778/边际77.712/Mixup78.824/组合77.059%。四新候选均FAIL，Mixup+1.176pp但2/3正，组合相对Mixup3/3负。80hash/72指标组完成后复核通过；RF概率边际接近均匀。保留原基线，下一建议逐级卷积生成器×Transformer/mean-MLP头2×2（尚未执行），未来关闭。
当前源期RF蒸馏对照：`20261002T022817Z_source_rf_distillation_c7b02e5a`。9新+3历史完成；CE77.647/KD0.1 78.039/KD0.5 78.627/打乱78.824%。真实KD均FAIL；打乱+1.176pp、3/3正/F1+1.351pp过数值门槛，但不支持样本知识迁移，列软目标正则化开发候选，不自动采用随机教师设计。完成后59hash/48指标组及初始化/索引/选模复核通过，12/12末步低于best。下一建议常量教师边际分布/均匀目标对照尚未执行；距90%11.176pp，未来关闭。
当前TAM生成器归一化层级对照：`20261002T013714Z_tam_generator_norm_hierarchy_f3920b88`。9新+3历史完成；accuracy flat_none77.647/flat_norm78.497/hier_none77.255/hier_norm78.627%。全部增量未过门槛；最高+0.980pp、2/3正，保留77.647%基线，78.627%仅观察候选（距90%11.373pp）。完成后49hash及48指标组/初始化/索引/选模复核通过。既有RF/current预测同valid行/TAM/标签核验：baseline三seed共同错53条中RF三seed全对22条，仅诊断不作oracle路由。下一候选source-only RF蒸馏对照尚未执行，未来关闭。
当前TAM输入配方对照：`20261001T235245Z_tam_input_recipe_factorial_ba217e20`。9项新训练+3项mask历史复用完成；log_current77.647/raw_current74.118/log_rfstyle70.196/raw_rfstyle69.542%。所有新候选对原方案3/3seed下降，不采用。完成后53冻结hash及48预测指标组、初始化/索引/选模复核通过。保留log1p+原AdamW配方+多尺度生成器+Transformer+mask77.647%（距90%12.353pp）。RF输入/优化设置移植未复现其优势，单一瓶颈未定位；下一候选是当前TAM生成器的局部归一化或逐级编码对照，尚未执行，未来关闭。
当前GPU卷积分类器对照：`20261001T154039Z_gpu_tam_temporal_cnn_b0a9af83`。3项GPU CNN完成、3项GPU Transformer基线历史复用；accuracy73.203%/F1 72.371% vs baseline75.948%/75.295%，差−2.745/−2.925pp，3/3seed下降，不采用本版卷积替代。完成后本run41及前run67冻结hash、24本轮+12历史mask指标组复核通过。当前自有框架候选保持多尺度生成器+Transformer+mask77.647%（距90%12.353pp）。下一步建议先审计与历史RF参考的编码/配方差异再设计有界对照，未新增训练；未来关闭。
当前TAM并行方向实验：`20261001T143214Z_parallel_tam_accuracy_directions_b989a1ab`。用户授权CPU转GPU，原队列stopped，18GPU+1CPU完成及CPU部分checkpoint保留。GPU仅mask77.647%（+1.699pp、3/3正）过门槛。CPU结构对照未完成、不作结论；替代GPU三seed对照见顶部，旧产物和结果暴露完整保留。
当前TAM生成器深度容量实验：`20261001T132328Z_gpu_tam_generator_depth_width_d4bfafde`。9项新训练完成、3项2层16通道多尺度基线历史复用；valid accuracy d2_w16=75.948%/d2_w32=76.863%/d4_w16=76.209%/d4_w32=75.817%。完成后57文件hash、48组预测指标及选模日程复核通过。宽32最高但+0.915pp、2/3正，全部增量未过预定门槛；保留d2_w16开发基线，停止本轮点位残差加深/加宽扩张。最高距90%仍13.137pp；末轮source接近100%、valid全部低于best，无延长训练依据。下一候选为固定生成器结构与输入的Transformer/保序卷积分类器对照，尚未执行；未来及漂移调整关闭。
当前TAM多尺度生成器实验：`20261001T124113Z_gpu_tam_multiscale_generator_101ab9e9`。6项新训练完成、3项原TAM编码历史复用；valid accuracy local73.333/context73.987/multi75.948%。multi−local+2.614pp、3/3正，过门槛；multi−context+1.961pp、2/3正，未过门槛。56文件hash及36组指标完成后复核通过；multi为新开发候选，90%仍差14.052pp。后续固定尺度的生成器深度/容量对照已完成（见顶部）；未来关闭。
当前生成器视角消融实验：`20261001T113740Z_gpu_generator_view_ablation_35fd3e98`。9项新训练完成、3项fusion历史复用；valid accuracy方向56.405/原packet70.523/TAM73.333/fusion74.183%。fusion−TAM+0.850pp、2/3正，未过门槛；packet时间增量+14.118pp过门槛。55文件冻结hash及48组指标完成后复核通过。后续TAM生成器编码受控比较已完成（见顶部）；90%未达，未来关闭。
当前多视角生成器×分类器实验：`20261001T104959Z_gpu_multiview_generator_head_factorial_cf47d62c`。12任务完成；A/B/C/D valid accuracy=63.203/67.582/66.797/74.183%；生成器D−B+6.601pp、Transformer D−C+7.386pp，均3/3seed正。D source99.776%，90%目标未达；保留组合候选，下一候选为本版单视角/融合归因。完成后冻结hash与48组预测指标复核通过；无新训练或未来评价。
当前研究目标（2026-10-01用户明确）：先设计自有生成器+分类模型，再针对该系统设计缓解时间漂移的调整机制。基础框架可按证据切换，Transformer不是必须保留；基础源期accuracy三seed平均≥90%仍为目标。新多视角2×2实验只检查基础编码/分类器，不自动进入未来日期评分或漂移适应。
当前原生基线实验：`20261001T091714Z_gpu_native_varcnn_rf_source_ce09ee6b`。6任务完成；valid accuracy varcnn=68.497%/rf=85.948%；12份best预测重载与独立指标核验通过。
当前强基线审计：`20261001T064615Z_wf_native_multimodel_audit_912c432f`。DF/RF/Var-CNN/ARES四模型源码审计完成，无训练或新数据访问。优先官方Var-CNN纯方向配置；RF需单列方向+时间TAM并审计源期接口；DF复用70.784% accuracy；ARES10000包及随机eval问题待解。目标明确为固定源期valid三seed平均accuracy≥90%，不保证可达；后续草案见run/NEXT_EXPERIMENT.md。
当前分段汇聚实验：`20261001T061340Z_gpu_packet_segment_readout_a42dcd83`。6任务完成；valid F1 historical_mean=73.394%/global_repeat=71.885%/segments=72.394%；12份新预测与6份历史预测指标核验通过。
当前样本量遮挡实验：`20261001T053149Z_gpu_source_size_span_mask_812f0250`。9任务完成；valid F1 s150_clean=71.416%/s150_mask=73.394%/all_clean=72.795%/all_mask=72.982%；18份新预测与6份历史预测指标核验通过。
当前源期审计：`20261001T052912Z_source_error_capacity_audit_b32edd4f`。完成：可用18553条，新增3253；每类156–190；MLP三seed共同错83/510；6份历史checkpoint预测核验一致，未来关闭。
当前渐进packet实验：`20261001T044328Z_gpu_progressive_packet_transformer_b7ba43ae`。6任务完成；CNN-Transformer/MLP valid F1=70.994/71.416%；重载核验12份新预测、历史指标核验12份通过。
当前GPU源期诊断：`20261001T041301Z_gpu_source_fit_native_baseline_ee3c7640`。小集99%拟合门槛=True；DF/Transformer valid F1=69.643/50.881%；DF相对差距门槛=True；9任务完成、重载预测及独立指标核验通过。

当前分离学习率实验：`20260928T114412Z_source150_separate_learning_rates_944201ea`。分离学习率完成：A/B/C/D valid F1 52.440/49.369/48.348/42.028%；通过比较[]；48/48预测核验。

当前有序残差实验：`20260928T015718Z_source150_ordered_token_residual_17a21a31`。150条有序残差完成：A/B/C valid accuracy 54.248/48.627/48.301%，F1 52.440/46.917/46.397%；有序候选PASS=False；36/36新预测重载、18/18历史指标复核。 两个新条件训练与验证均低于基线，不采用该残差接法；不能据此证明局部顺序信息无用或唯一归因于过拟合。当前最好仍为150条/类+1层无残差，valid accuracy54.248%、F1 52.440%；Day14未访问。

当前150条锚定×dropout实验：`20260927T091205Z_source150_generator_anchor_dropout_5b9c49f3`。150条锚定×dropout完成：R00/R10/R01/R11 valid F1 52.440/50.385/46.447/46.798%；通过比较[]；72/72预测核验（含18历史复核）。

当前优先级修正（2026-09-27）：用户明确54.248%源期验证准确率尚不满足要求，继续提升基础模型，暂不进行Day14评价。B150×1层仅为当前开发对照，不视作达标模型。复查三seed曲线：best为10880/11360步，12800末步valid F1均未超过best；不把增加步数视为必然提升，也未证明唯一瓶颈。下一候选有两类：生成器信息保留，以及固定150条/1层下的温和优化约束。后者已获用户授权，见顶部当前实验（初始化锚定lambda0.001 × dropout0.1/0.15）；这是较小改动的拟议筛查，不是瓶颈已定位。保序残差作为后续表示候选，历史固定保序表示阴性仍有效。此处是设计来源记录，执行状态以上述当前实验为准；未来评分仍关闭。

当前样本规模×深度实验：`20260927T032819Z_source_size_local_depth_factorial_dae714b7`。80/150条×1/2层完成：A/B/C/D valid F1 46.939/52.440/44.567/51.949%；通过比较['B_150_l1−A_80_l1', 'D_150_l2−C_80_l2', 'D_150_l2−A_80_l1']；72/72预测核验（含18历史复核）。

Proteus 文献核对（2026-09-27）：已核对本地《Enhancing Website Fingerprinting Attacks against Traffic Drift》原文第7–9页及附录A，并目视检查图5。图5准确率与表III的P/R/F1均从Day14开始，未找到明确的Day0评价数值；附录A将2024-03-13定义为训练流量采集Day0。表III未加Proteus基线的Day14 F1：BAPM 62.08%、ARES 68.82%、DF 71.94%、Tik-Tok 77.89%、Var-CNN 79.77%、RF 87.63%；这些不能标为Day0准确率，也不能直接等同本项目固定valid Macro-F1。论文说明使用官方基线代码、调参及5折交叉验证，精确训练样本预算和源期留出规则仍需核对。当前仅核对文献，未启动训练或未来评分。

当前充分训练实验：`20260927T011115Z_source80_extended_training_b7d4d924`。80条充分训练完成：CLS/mean valid F1 46.939/43.393%；共同提高门槛{'C_80_cls': True, 'D_80_mean': True}；36/36预测核验。

性能目标修正（2026-09-27）：用户明确要求与已有WF模型各自配套的原生输入和训练流程达到的准确率比较，不能把输入不适配当前方向/时间表示的DF结果当作当前模型性能目标。历史DF数值仅作架构不匹配的背景，不作准入门槛。后续应先审计同数据、同类别、同标签权限和同评价角色的原生WF基线，再冻结当前Transformer的源期与时间漂移比较目标。

当前样本规模与汇聚实验：`20260926T155907Z_sample_size_token_readout_6d47208d`。样本规模×汇聚完成：A/B/C/D valid F1 21.898/21.271/35.848/37.644%；通过比较['C_80_cls−A_20_cls', 'D_80_mean−C_80_cls', 'D_80_mean−B_20_mean', 'D_80_mean−A_20_cls']；36/36预测核验。

当前正则化实验（2026-09-26）：`20260926T145102Z_generator_classifier_regularization_1f0406ee`。2×2正则化完成：R00/R01/R10/R11 valid F1 21.329/15.062/23.029/16.359%；通过比较['R11_both−R01_classifier_reg']；24/24预测与约束核验通过。 无新条件稳定胜过R00；唯一PASS仅为R11相对受损R01。强分类器正则化不采用；生成器锚定+1.700pp但2/3seed正，仅保留候选，未追加实验。

当前固定生成器重训（2026-09-26）：`20260926T081359Z_frozen_generator_classifier_retrain_ccaff3b4`。固定B/C生成器重训完成：INCONCLUSIVE；B−C valid F1 +0.433pp，B恢复 +1.879pp；12/12新预测重载核验。

当前归因诊断（2026-09-26）：`20260926T075128Z_generator_classifier_diagnostic_52a460a4`。C/B/D冻结token诊断完成：INCONCLUSIVE；B−C packet F1 +1.202pp，B all_views探针−原模型 -7.594pp；18原预测/108探针预测/54拟合核验通过。

诊断解释及后续完成情况：B−C packet探针source F1 +18.869pp，valid仅+1.202pp且正则敏感性方向不稳，提示训练集偏向，但尚不能归因于生成器单独过拟合；线性探针也未超过原分类器，未证明分类器利用不足。此前建议的固定B/C生成器、配对重训非线性分类器现已完成（见顶部记录），故更正“尚未启动”状态：新B−新C valid F1 +0.433pp，逐seed两正一负；B相对原模型恢复+1.879pp，亦两正一负，未过一致性门槛。训练集F1差+17.018pp，训练集偏向迹象仍在，唯一归因未决。线性探针有效结果为v2，CPU推理数值修正及首版作废记录见该run的PLAN.md/RESULTS.md。

当前并行实验设计（2026-09-26）：`20260926T040212Z_classifier_generator_parallel_946c81c4`。CPU生成器四条件完成：A/B/C/D valid F1 23.104/23.229/24.979/24.520%；24/24预测与反馈边界核验通过。

当前设计要求（2026-09-26）：用户明确当前只提升基础分类器；生成器也须通过分类训练反馈学习特征→token转换。已核对：现有统计提取/patch构造为固定规则，模型内线性投影已随分类损失训练；不能把已有投影改名就称为新增生成器学习机制。下一版拟把可学习局部编码和汇聚置于原始方向细节被压缩之前，作为生成器参数与分类器端到端联合训练；首轮仍固定输入预算、token数及run/window边界，暂不同时学习离散切分。反馈首先定义为source分类损失梯度，valid只按预定规则选模/评价，不进入梯度。建议配套冻结/可训练生成器对照并保留强摘要基线，不能仅凭参数更新或训练准确率提高判为有效。本次仅登记设计要求与拟议实现，未新增训练；现有CPU保序实验阴性结论不变。

当前CPU首轮实验（2026-09-26）：`20260926T024929Z_cpu_packet_ordered_representation_67b648d4`。CPU保序表示完成：summary/ordered valid F1 24.655/17.162%，差-7.493pp，0/3 seed正，FAIL_CANDIDATE；12/12重载核验通过。

最新收口（2026-09-26）：`20260925T161437Z_transformer_fullsource_capacity_cpu_0de7931a`。全部 source2040 标签监督训练，层次化三 seed×100 epochs 完成：valid accuracy 27.320%、Macro-F1 25.235%，比历史 DF-only 性能标尺低 21.830/22.909pp。平铺仅 seed1729 完成（valid 27.451%/25.854%），seed3407 在预定 5400 秒上限超时、seed2026 未开始；8/12 已产生预测及checkpoint复算通过，完整实验按预算停止，不作两结构稳定性比较。当前自有基础 Transformer 仍未接近成熟 DF 水准；下一步应单独预定保序 packet 表示/模型能力与计算可行性实验，暂不叠加预训练、蒸馏或漂移模块。

最新完成（2026-09-26）：`20260925T155155Z_transformer_hierarchical_backbone_cpu_e57e796c`。依据 CipherSight L1/L2 思路做视角内局部编码+跨视角全局分类，但只使用原 packet/run/window 摘要，非TLS record/flow复现，且无resource监督。固定5-shot、三seed、100epochs CPU，层次化参数113670相对平铺110400。train accuracy 53.20% vs 40.07%，valid accuracy 14.38% vs 15.69%，valid Macro-F1 12.50% vs 12.91%（−0.41pp，仅1/3 seed正）；预定主模型候选门槛失败。6/6预测/checkpoint核验0错，未访问未来日期/WTT/AWF，无预训练、蒸馏或TTA。旧平铺对照为明确复用的已观察结果。历史DF 48.14%是全部source标签训练，不可与本轮510条5-shot直接作同预算门槛。下一步先在全source标签和相同输入权限下建立主模型公平对照，或单独验证保序packet编码/解除run截断；不继续无界叠加模块。

最新完成（2026-09-25）：`20260925T135213Z_transformer_cpu_convergence_07cd94a5` CPU 100 epoch 收敛诊断，固定上一轮输入/模型/510 条 5-shot、三条件×三 seed。packet scratch/多视角 scratch/同域预训练 valid accuracy 15.16/15.69/16.93%，Macro-F1 11.81/12.91/14.84%；最佳 checkpoint 的 train accuracy 31.57/40.07/45.16%。25 轮确实过少，但 100 轮仍未充分拟合，且低于既往同 5-shot 轻量 CNN 约20.30% valid F1。预训练相对多视角 scratch +1.92pp、仅2/3 seed正；不称稳定增益或抗漂移。4/9 最佳 epoch 在100；18/18 预测、checkpoint、指标复算0错误。仅 source/valid，未来日期、WTT/AWF 和 TTA 未访问。下一步优先预算匹配地区分输入表示与优化/分类头瓶颈，不在同一 valid 无界延长训练。

最新完成（2026-09-25）：`20260925T130044Z_transformer_temporal_pretrain_fewshot_7098625e` 固定 TemporalDrift source/valid 和每类 5-shot，packet scratch/多视角 scratch/多视角同域无标签预训练三条件×三 seed、25 epochs。valid macro-F1 3.732/2.625/4.511%；预训练相对多视角 scratch +1.886pp、3/3 seed 正、五开发日期均值全部正，按预定相对候选门槛通过；相对 packet scratch valid +0.779pp、三 seed 正、五日期绝对 F1 亦高。但 Day270 只有 2.918%，valid→Day270 下降 1.593pp，不优于 packet scratch 的 1.554pp；绝对性能远低于既往同 5-shot 轻量 CNN valid 20.298%，不能称抗漂移或可用主模型。多数最佳 epoch 在 21–24/25、训练准确率低，需另立有界收敛检查，暂不在弱教师上启动 TTA。63 预测/checkpoint/指标重算 0 错误，future 仅开发评分，WTT/AWF 未访问。

最新完成（2026-09-25）：`20260925T124119Z_transformer_generator_prototype_ea2ff5b2` 完成固定 packet/run/window 生成器 + typed 小型 Transformer 原型。实现同一原始 packet span 在多视角中联合遮挡的无标签重建、少标签 CE 微调、零初始化 bottleneck adapter，以及冻结 teacher 的高置信 KL + 双视角一致性 adapter-only TTA。43 项测试运行，42 通过、1 项既有可选 parity 检查跳过；覆盖完整 5000 包预算、batch 独立位置、mask 不泄露、有限梯度和 TTA 主干/teacher 冻结。未读取真实数据、未训练分类模型、未访问未来日期，因此不代表 Transformer/蒸馏/TTA 已有效。真实实验前仍需冻结无标签权限、扰动、伪标签阈值、步数和回滚规则。

最新完成（2026-09-25）：`20260925T125021Z_transformer_real_smoke_0b6d2365` 用四条固定 TemporalDrift source trace 完成真实接口 smoke：span 重建、CE 微调和 adapter-only TTA 单步均通过；teacher 与 student 非 adapter 主干冻结隔离通过。未读取 valid/未来、未评分性能、未保存 checkpoint；TTA 阈值 0.8 本批选中 0 个 teacher 样本。该结果只证明真实输入接口闭环，不证明效果。

最新完成（2026-09-25）：`20260925T121122Z_pcap_ssl_three_way_c275715d` 首轮外部 PCAP 自监督预训练少标签筛查。140 个 PCAP/PCAPNG 只读解析出 465 条主 flow 序列；相同融合模型、TemporalDrift source 每类 5-shot、三 seed 下，scratch/PCAP packet 预训练/PCAP window 预训练 valid macro-F1 为 20.298/19.082/19.186%。两种预训练在三 seed valid 均未超过 scratch，五开发日期也无稳定收益；不能据此否定更大或同域无标签预训练。v1 因封存路径及 window 边界实现错误作废；v2 使用正式生成器重跑，63 checkpoint/预测/指标重算 0 错误，source/valid 及 PCAP 对目标集方向完全重复 0。PCAP 仅为跨应用、非网站数据，预训练目标是简单遮挡重建而非 ET-BERT 完整 MBM/SBP；WTT/AWF 未访问。下一步不在同一 valid 上无界追调；先审计更合适的同域无标签数据和预训练目标，再预定一轮对照。

最新完成（2026-09-23）：`20260923T122727Z_gpu_varcnn_window_fusion_f65c399f` 使用本项目已有 VarCNNDirection（原 VarCNN 方向编码器的明确派生版）对比原始模型/常量window融合/真实window融合，三seed×45epochs。valid macro-F1 27.190/16.137/18.107%；真实相对常量+1.971pp（2/3 seed正）、五TemporalDrift开发日期均值全部正，token归因候选门槛通过；但真实相对原始 VarCNNDirection valid−9.082pp、五日期全部负，部署准入失败。融合头本身严重损伤原始主干；该接法不适合作第一版。source/valid方向重叠0、63 checkpoint预测/指标及输入统计独立核验0错误。TemporalDrift仅开发，WTT/AWF未访问。不要把此阴性结果推广为所有VarCNN融合均失败；若继续须保留原分类路径并预定超过原始模型的门槛，不在同一valid无界调参。

最新完成（2026-09-23）：`20260923T104034Z_gpu_df_token_logit_residual_5a92bbaa` 保留原DF分类头，生成token仅做零初始化类别分数残差；原始DF/常量残差/真实残差×三seed×45epochs。valid macro-F1 48.144/45.212/45.354%；真实相对常量仅+0.142pp（2/3 seed正）、五TemporalDrift开发日期均值差均正，按预定弱归因门槛通过；但相对原始DF valid −2.790pp、三seed全负，五日期全部负，部署准入明确失败。输入/统计、source/valid方向重叠0、63 checkpoint预测与指标重算核验0错误。此前局部特征残差也只恢复常量分支损失而未胜DF；不要在同一小valid继续追调结构或把弱token增量说成成熟模型收益。TemporalDrift仅开发，WTT/AWF未访问；外部评价前仍须冻结合格候选并审计权限。

最新完成（2026-09-23）：`20260923T102339Z_gpu_df_local_token_fusion_112b50b9` 用 DF 专属保序双尺度局部 token 编码（100/20位置→DF 18位置特征图）及零初始化残差，三条件×三 seed×45 epochs。valid macro-F1 原始 DF/局部常量/局部真实 token 为46.264/44.895/46.110%；真实 token 相对同结构常量+1.216pp、三 seed全正，Day14/30/90/150/270差+0.884/+0.384/+0.716/+0.144/−0.197pp，预定 token 归因候选门槛通过。但真实 token 相对原始 DF valid −0.154pp，未来仅Day150微正，Day270 −0.562pp，尚无成熟 DF 的部署收益，不能称抗漂移。输入/抽样/标准化、63 checkpoint预测及指标核验0错误。上一轮扁平末端融合阴性与本轮局部融合正向归因共同说明 DF 集成方式重要，但两轮改变不止一个因素；TemporalDrift 仍仅开发，WTT/AWF 未访问。后续需以胜过原始 DF 为显式准入，避免同一 valid 反复调结构。

最新完成（2026-09-23）：`20260923T094615Z_gpu_df_window_fusion_4d785e2e` 按用户要求在更强 DF 主模型上直接使用生成 window token，三条件（原始 DF、等结构常量分支、真实 window 分支）×三 seed×45 epochs。valid macro-F1 45.681/46.364/35.358%；真实 window 相对常量分支 −11.006pp，三 seed 全负；Day14/30/90/150/270 差均负（−8.441/−7.675/−7.408/−6.590/−6.409pp），预定 token 候选门槛失败。DF 主干初始权重一致、融合两条件参数一致；63预测与 checkpoint/指标重算、source-only 标准化、抽样和输入散列核验0错误。该阴性结果说明先前轻量 CNN 的正向结果未推广到当前 DF 集成方式；不能断言生成器普遍无效。真实 window 末轮训练拟合更高但 valid/未来更低，后续若研究须先预定优化/分支交互诊断，不在同一 valid 上事后追调。TemporalDrift 仍仅开发；WTT/AWF 未访问。

当前方向决定（2026-09-23）：用户停止将生成器限制为训练期的教师→packet 学生蒸馏路线，改为直接保留生成器 token 及专属处理层参与主模型训练与推理。所请求的首轮直接使用基线已由 `20260923T075107Z_gpu_packet_window_main_378247f0` 完成并核验：fusion 相对 packet-only、window-only 在 valid 和五个 TemporalDrift 开发日期的绝对 macro-F1 均值均更高。不要重复同配置训练；下一轮若推进，应另立明确问题和公平对照（如更强 packet 基线或冻结后的外部评价），不能把现有直接融合收益称为抗漂移效果。

最新完成（2026-09-23）：`20260923T090922Z_gpu_class_relation_distill_69bae911` 冻结融合教师 source 类别原型关系，比较 packet-only 学生 CE、同教师 packet 关系与 window 关系，三 seed×45 epochs。valid macro-F1 24.466/26.385/25.120%；window 相对 CE +0.654pp、三 seed 均正，但相对 packet 关系 −1.265pp，仅 1/3 seed 正；Day14/30/90/150/270 相对 packet 关系均值全部负，预定 window 特异门槛未过。教师 window source 关系真类 top-1 约34–35%，高于 packet 的20.3%，但该信号未有效转移为 packet-only 泛化收益。18 关系数组、63 预测/指标和模型重载、抽样及散列独立核验 0 错误。TemporalDrift 仅为已观察机制开发；WTT/AWF 未访问。不要据此称生成器训练期独用成功；下一步先厘清目标可转移性与部署定位，避免在同一 valid 追调权重。

最新完成（2026-09-23）：`20260923T090023Z_frozen_feature_transfer_probe_18dd81c9` 冻结上一轮融合教师与普通 CE packet 学生，用固定 source/valid 做类别判别力与线性可转移性诊断；无未来日期或外部数据访问。教师 window/packet 真实特征 ridge 探针 valid F1 23.880/18.044%，window 三 seed 均较高，说明目标有类别信号。冻结 CE 学生特征线性重建的 valid R² window/packet 0.714/0.790；同一探针在重建特征上的 F1 为 14.745/16.137%，相对真实特征下降 9.135/1.907pp。窗口特征总体可部分重建，但类别关键细节明显丢失；不支持“特征本身无用”，也不能证明非线性端到端学生无法转移。18 预测、36 特征/统计、36 探针与重建器参数复算及散列 0 错误；教师/学生锚点和 source-only 标准化核验通过。若训练期独用继续，需预定更类别相关的转移目标与同教师 packet 特征控制，避免同一 valid 上追调权重。

最新完成（2026-09-23）：`20260923T082606Z_gpu_feature_distill_b4f0ceb2` GPU 融合教师中间特征蒸馏。教师 valid F1 三 seed 均值 33.112%；同一 packet-only 学生 CE/对齐教师packet特征/对齐教师window特征 valid F1 24.873/25.046/25.296%。window 对齐相对 CE valid +0.423pp 但仅 1/3 seed 正；五未来日期相对 CE 均值小幅正，相对同教师 packet 特征对齐却全部负（-0.209/-0.598/-0.267/-0.030/-0.345pp），未过 window 特异收益门槛。第45轮 window 特征 MSE≈0.31、packet特征≈0.15，提示学生拟合 window 表示较难。63 学生预测和模型重载、18 教师 source target/统计核验 0 错误；WTT/AWF 未访问。训练期独用的 logit KD 与本轮固定特征 MSE 均未证明 token 特异增益；当前可靠正向仍为推理保留 token 的融合模型。后续应避免在同一小 valid 连续调 KD 权重，先明确部署定位/更强 packet 学生和预定机制，再考虑外部冻结评价。

最新完成（2026-09-23）：`20260923T080156Z_gpu_train_only_token_distill_548c9e47` GPU 训练期 token 蒸馏实验。两同结构教师三 seed×45 epochs：fusion 教师 valid F1 33.131%，packet 教师 24.893%；三同结构 packet-only 学生（CE/packet KD/fusion KD）valid F1 24.666/24.067/24.208%。fusion KD−CE valid -0.458pp（1/3 seed 正），五未来日期 +0.247/+0.079/+0.053/-0.322/+0.710pp；相对 packet KD 五日期平均 -0.198pp，仅 2/5 日期正，未过预定门槛。63 学生预测及 checkpoint、6 教师 source logits 重载核验 0 错误；WTT/AWF 未访问。结论：推理保留 token 的上一轮融合增益成立于该部署形态，但固定 T=2、权重0.5 的训练期独用 logit 蒸馏未证明增益。用户若要求训练期独用，下一机制应是预先限定的结构性蒸馏而非事后在同一 valid 追调权重；漂移后 adapter 仍未验证。

最新完成（2026-09-23）：`20260923T075107Z_gpu_packet_window_main_378247f0` GPU 主模型三条件（packet-only/window-only/fusion）×3 seed×45 epochs，固定 TemporalDrift 抽样，等总参数 219494、同初始化和预算。valid macro-F1 24.999/30.284/33.348%；fusion 相对 window-only +3.064pp、三 seed 全正，Day14/30/90/150/270 +1.982/+1.481/+1.096/+0.838/+1.049pp，13/15 seed×日期正；相对 packet-only 全日期更高。63 预测/指标/checkpoint 重载与抽样核验 0 错误。首次较直接支持生成 window token 专属层能帮助当前主模型训练与分类泛化；但 fusion valid→Day270 降 14.406pp，比 window-only 的 12.391pp 更大，不能称抗漂移。此轮 token 分支训练与推理均保留，未验证训练期独用/蒸馏或漂移后 adapter 更新；TemporalDrift 是开发证据，外部 WTT/AWF 未访问。下一步按部署定位选择训练期蒸馏对照或冻结候选机制并审计外部评价资格。

最新完成（2026-09-23）：`20260923T074303Z_gpu_tail_neutral_convergence_6e4e1ab5` 使用 GPU 0，固定 TemporalDrift 抽样和同一 full-source 标准化，full/tail-neutral 各三 seed×45 epochs。valid macro-F1 29.192→29.516%（+0.324pp，2/3 seed 正）；Day14/30/90/150/270 差 +1.109/+0.625/-0.033/+0.608/+0.408pp，4/5 日期正。valid→Day270 下降仅缩小约 0.084pp，不能称实质抗漂移。42 预测、42 checkpoint 重载重算、指标/选模/抽样核验 0 错误；WTT-Time/AWF 未访问。tail-neutral 保留候选，但下一关键 GPU 实验应转向容量与预算匹配的 packet 主模型和 token 专属处理层开关对照，验证生成器能否真正改善主模型训练及跨日期泛化；适配器在线更新仍须另立隔离协议。

最新完成（2026-09-23）：`20260923T045507Z_cpu_tail_neutral_matched_564216a8` 对上一轮 tail-neutral 做严格复核，full/neutral 两条件共享 full source 标准化统计、模型容量、15 epochs 和三 seed。tail-neutral valid macro-F1 +0.637pp；Day14/30/90/150/270 分别 +0.889/-0.150/+0.265/+0.494/+0.538pp，4/5 日期为正。支持将 tail-neutral 保留为候选 window 基线，但幅度小、仍属 TemporalDrift 开发证据，不是抗漂移证明；adapter 方向继续暂缓。

最新完成（2026-09-23）：`20260923T043131Z_cpu_window_tail_adapter_12d25c24` 固定复用 TemporalDrift 多视角筛查的 source/valid/Day14/30/90/150/270 sampling manifest，window_full、零初始化 window_adapter、window_tail_neutral 三条件×3 seed×15 epochs CPU。valid macro-F1 均值 25.363/24.084/26.125%；adapter 相对 full -1.279pp，Day14/30/90/150/270 分别 -0.582/-1.705/-0.415/-0.440/-0.057pp，未通过泛化门槛。tail-neutral 相对 full valid +0.762pp，未来 +0.836/+0.305/+0.260/+0.212/+0.611pp；仅说明 partial/tail 特征可能带噪，属于诊断性消融，不能称抗漂移方法。适配器当前应暂不保留；下一步若继续 CPU，应优先做 tail-neutral 的严格匹配复核（固定相同标准化与容量）或停止在该方向继续调参，而不是扩大适配器。

已完成（2026-09-22）：run 20260922T161534Z_cpu_multiview_convergence_593f69e9统一预算检查，继承run/window五条件，从头三seed各45epochs；结果详见该 run 的 RESULTS.md。旧“进行中”状态在 2026-09-23 核对登记后更正。

最新完成（2026-09-22）：run 20260922T114658Z_cpu_run_window_adapters_v2_148f23cd，run/window五条件×3seed15epochs CPU。run_pair/window_pair/fusion/shared/specific accuracy22.88/28.56/27.45/27.58/27.71%，F1 21.02/26.35/25.69/26.02/26.10%。specific-fusion F1三个seed均正、均值+0.411pp；specific-shared仅+0.079pp，两正一负，达到本轮宽松候选门槛但不能证明专属机制优越。fusion-window F1-0.657pp、specific-window -0.246pp，均两负一正，互补性门槛未通过。保留window强基线与适配器候选，不默认融合优于最佳单视角。参数差约3.1%，无严格等容量/等算力主张。15指标7650预测、2550输入独立重建、source-only标准化、封存/选模/重载核验0错误，35测试通过1跳过，3分20.43秒CPU，无GPU/未来/WTT/AWF。7/15最佳为末epoch，后续优先预定统一更充分预算检查收敛，尚未执行。前版20260922T114213Z_cpu_run_window_adapters_faf3c613首个前向维度错误，未更新参数/评分，failed记录保留。

最新完成（2026-09-22）：run 20260922T112458Z_cpu_token_adapter_v1_4a18e272，第一步“冻结生成器＋轻量训练适配器”实验。coarse run token固定生成，baseline/零初始化残差适配器/随机初始化残差适配器×3seed、15epochs、CPU3线程，source2040/valid510，未来/WTT/AWF关闭。accuracy22.94/23.07/22.48%，F1 20.87/20.90/20.60%；zero相对baseline平均F1仅+0.036pp（-0.301/+0.629/-0.221pp），未通过稳定收益门槛；random平均-0.270pp。适配器能从恒等初始化学习但本轮未改善泛化，不默认加入。训练2分08.31秒、9指标4590预测，封存/隔离/重载/选模核验通过；35测试通过1跳过。此轮是生成器冻结、主模型与适配器联合训练，不是在线漂移适配或训练期蒸馏；后续需run/window分支匹配实验或预先固定预算收敛检查。

最新完成（2026-09-22）：run 20260922T111009Z_cpu_exact_increment_control_d4ccdda7补齐精确长度归因，四条件×3seed15epochs CPU。bucket+MLP/常量分支/桶下界代理/真实精确值 accuracy22.94/23.07/23.14/23.27%，F1 20.87/21.04/21.19/21.26%。严格同参数同处理exact-proxy F1仅+0.071pp、两正一负；exact-bucketMLP+0.389pp亦两正一负，均未通过一致门槛。没有精确长度稳定独立增益证据，不是等价/无用证明。proxy-bucketMLP+0.318pp三seed正但幅度小。建议以bucket+MLP为简洁基线，不继续在同valid追逐微小调参；后续可有界增加同等训练预算检查收敛或检验run/window互补性，尚未执行。12指标6120预测封存/选模/重载/隔离核验通过，exact锚点1530预测复现前轮；35测试通过1跳过。source2040/valid510原清单，CPU3线程，无GPU及未来/WTT/AWF访问，无训练期蒸馏/漂移验证。

最新完成（2026-09-22）：并行CPU两项验证完成。诊断run 20260922T064229Z_cpu_token_branch_diagnostic_aa17da3e只读上轮hybrid：full/bias_only/off valid F1 19.63/16.98/7.94%，删除连续变量三seed均下降，变量RMS仅bucket约6.2–11.2%，不支持简单有害干扰/幅度压制解释；训练后移除是分布外干预。训练run 20260922T064218Z_cpu_token_gated_fusion_24ddd51d五条件×3seed15epochs，bucket/hybrid/separate/fixed0.1/gate accuracy22.09/22.29/22.75/23.27/23.01%，F1 20.04/19.61/20.52/21.26/20.98%。separate-hybrid F1三seed均正均值+0.91pp；gate-hybrid+1.37pp、gate-bucket+0.94pp均通过开发筛查，但gate-fixed三seed均负均值-0.28pp，未证明学习门控增益。支持独立处理/受控融合候选，不证明精确值独立增量；下一步优先补仅bucket+同一MLP匹配对照，尚未执行。诊断18指标22950预测、训练15指标7650预测及封存核验通过；训练4分06.52秒CPU3线程、诊断9.77秒1线程，无GPU/未来/WTT/AWF访问。两方案预先独立冻结。无漂移/训练期蒸馏验证，旧状态中的window扩展优先级被此次机制诊断更新。

最新完成（2026-09-22）：run 20260922T062732Z_cpu_token_typed_encoding_df081d9e，CPU长度专属编码continuous/bucket/hybrid/hybrid_norm×3seed，source2040/valid510、15epochs。accuracy均值5.69/22.22/22.35/21.50%，macro-F1 2.47/20.39/19.63/18.96%。hybrid-continuous三seed均正，平均accuracy+16.67pp、F1+17.17pp，支持编码重要；hybrid-bucket F1两负一正、平均-0.75pp，无精确分支稳定增量；norm亦无增益。保留bucket简洁基线，建议下轮容量受控run/window专属分支互补性消融，尚未执行。12指标6120预测、封存/隔离/全输入重建/选模/checkpoint重载核验通过；35测试通过1跳过，2分20.77秒CPU，无GPU及未来/WTT/AWF访问。此轮训练/评价保留token encoder，不证明训练期蒸馏或抗漂移；下文文献核对阶段“未实施”已由本轮编码试验更新。

当前设计要求（2026-09-22）：用户希望生成token有专属特征处理层，再用于模型训练，而非只作为packet辅助回归目标。已核对CipherSight原文 https://arxiv.org/html/2608.13905v1 的Representation and Hierarchical Encoding及P1/P2章节：类别字段embedding，数值字段log bucket embedding+log值线性投影，求和归一化，再做流内/流间编码；资源监督及教师仅训练使用。可借鉴类型感知编码与结构建模，不把本项目run当TLS record/resource，不凭空构建flow身份或资源标签。建议下一实验优先在相同输入、近似容量、相同source/valid及预算下比较现有编码与类型专属编码，exact增加bucket+continuous组合对照；窗口分支融合另作消融，避免混淆新增信息与编码收益。当前仅完成文献核对和设计澄清，未实施该新层或新增训练。训练期token分支是否推理保留仍须显式界定：若只训练使用，需蒸馏到部署分支，不能直接删除直接token分类模型的输入分支。漂移缓解仍为其他模块职责。

当前目标澄清及直接token结果（2026-09-22）：用户明确希望用生成器输出token直接训练主模型；此前辅助回归是助手提出的局部方案，不是用户最终要求。run 20260922T060351Z_cpu_token_native_classifier_6463ceb7完成方向/长度/位置编码+保序CNN+分段mask池化，训练推理均仅token。三seed source2040/valid510，coarse accuracy23.01±0.97%、macro-F1 20.49±1.49%；exact连续长度投影6.14±0.63%、2.61±0.30%。支持coarse直接学习流程；编码方式不同不能推断粗化优于精确。6指标3060预测核验0错误，63.10秒CPU，无未来数据评分。下一步优先诊断exact长度编码，并做匹配架构输入对照；生成器不以减缓漂移作为准入要求。下文辅助训练定位已被本次澄清更新。

当前定位及最新结果（2026-09-22）：用户明确生成器仅辅助训练，漂移缓解由其他模块承担。此前直接输入window的正向结果不等于训练辅助有效。run 20260922T055557Z_cpu_generator_aux_training_68a9328c完成CPU三条件×三seed：同一packet推理模型，baseline/window辅助/run辅助 valid accuracy均值23.79/23.86/23.79%，macro-F1 20.79/20.96/20.85%；两辅助均有负seed，未通过一致收益门槛。固定CE+0.1*MSE、15epochs、source2040，结论仅针对该辅助配置。9指标4590预测完整性0错误，无GPU、未访问未来日期。下一步先诊断辅助损失与主任务学习，再确定有界后续方案。

最新TemporalDrift CPU结果（2026-09-22）：v1 `20260922T023317Z_cpu_temporal_multiview_screen_639ccab3` 因固定source/valid有2条方向重复，在训练前按规则STOP。v2 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 排除完整valid重叠并source canonical去重后完成四视角×3seed。windows在Day14/30/90/150/270相对packet平均+3.45/+2.76/+2.73/+2.79/+1.45pp，15/15 seed-date配对为正，5/5通过候选门槛；valid/window 27.84%、packet23.53%。但valid到Day270下降windows12.45pp、packet9.59pp，不能称减小漂移衰减。exact/coarse五日期均低于packet。72指标、128520预测、方向交叉/封存/选模核验0错误，无GPU。下一步做容量受控packet+window增量对照。

最新CPU结论（2026-09-22）：`20260922T022343Z_cpu_run_order_ablation_5c3b1b73` 同容量run顺序消融完成，exact/coarse×原序/固定乱序×3训练seed。原序相对乱序accuracy在18/18个视角-角色-seed单元为正；均值差exact valid/JP/subpage +4.18/+3.42/+2.06pp，coarse +3.92/+2.10/+1.94pp，macro-F1亦18/18为正。预注册valid+JP一致门槛两视角均通过。36指标、55032预测完整性0错误，2分56.83秒CPU，无GPU。支持run排列为有效信息和保序编码；固定单一shuffle、JP/subpage非时间日期，不能声称抗时间漂移。下一步核对并在TemporalDrift日期上做CPU多视角开发评价。

最新CPU进展（2026-09-22）：`20260922T021205Z_cpu_ordered_run_encoder_c6048d5a` 两层局部run卷积完成，同样本、1seed、15epochs。exact/coarse valid accuracy12.55/10.78%，JP10.27/8.45%，subpage6.13/4.80%。均高于前序池化读出，但容量亦增加，不能将收益单独归因于顺序。6指标、9172预测、抽样/封存/选模核验0错误；54.74秒CPU，GPU隐藏。下一步可做同容量顺序消融与多seed，不据此冻结最终方法或宣称抗时间漂移。

当前事项（2026-09-22）：用户授权CPU可执行部分。`20260922T015431Z_cpu_tiny_multiview_learning_2a2e1036` 已完成四模型、1seed、15epochs CPU学习筛查，沿用2040/510/2036/2040样本。packet/window valid accuracy 26.67/29.80%，JP21.46/23.23%，subpage11.72/11.03%；exact/coarse为弱mean/max token读出，valid5.88/4.90%。12指标、18344预测、抽样、封存及选模独立核验0错误。37.26秒、无GPU。支持packet/window可学习信号，不能证明生成器增益或抗时间漂移；架构/参数量不匹配，3模型最佳为末轮，未证明收敛。下一步可继续CPU保序run编码与更充分source训练，不能据弱读出否定run或宣称只能等待GPU。下文为历史事项，未完成/尚无训练等描述只对应当时阶段。

最新效用开发试验（2026-09-22）：`20260922T011309Z_token_effect_diagnostic_d0b9386f` CPU六系统类原型试验完成，source2040/valid510/JP2036/subpage2040，一个seed。源packet accuracy仅11.57%，读出较弱；windows在JP16.45%、subpage8.24%，packet10.36%/6.52%；全视角等权9.82%/4.26%。不能据此否定token学习或证明分级路由必要。随后完成 `20260922T012712Z_multiview_cpu_structure_audit_1c504ca7`：41466行X-only审计，内部零0；packet预算1000截断约50–56%，run槽位与原始包位置显著漂移。完成 `20260922T013510Z_multiview_batch_adapter_69a56f75`：packet/exact/coarse/window显式mask与截断接口，全项目36 tests通过1跳过。完成 `20260922T014216Z_cpu_learned_readout_probe_7db3ce59` 修订版：固定CPU线性读出packet valid11.57%、windows20.20%；JP9.04/16.94%、subpage4.85/8.19%，与原型接近，无稳定learned增益；v0误加window长度列已作废保留。JP/subpage已用于性能开发，其他条件未评分。无GPU训练，外部WTT/AWF保持关闭。下一步需GPU空闲后冻结容量匹配的深度baseline，不自动设计selector。

当前候选设想（2026-09-22，用户确认）：研究分级释放不同特征token的机制，面向不同漂移情况考察不同视角的效用。尚未实现分级选择器，未证明能从部署可观测信息识别适合释放的视角；不得默认使用真实漂移原因/目标类别作为路由输入。现有多视角生成器只是候选特征接口，不是已验证适应机制。本次仓库整理仅本地Git提交，不训练或推送远端。

当前实现事项（2026-09-22）：多视角表示生成器v0已完成，run `20260922T005618Z_multiview_feature_generator_0b5365cf`。traffic_views.py独立输出packet方向、exact/coarse run、局部方向窗口、显式可选同方向时间/run跨度、signed-size视角；字段边界见TRAFFIC_VIEWS.md。32项测试31通过1跳过，三场景384行真实接口检查通过。未训练评分、未读真实标签/URL、未开放WTT/AWF。尚未实现时间bin、batch collator、模型融合/蒸馏；下步需冻结视角对照和数据角色，不能把接口通过当抗漂移证据。下列旧实验结论保留。

状态：共享更新干扰存在性诊断 `exp_dce0488c23844cb7` 已完成，预注册裁决 `STOP_NO_USABLE_INTERFERENCE`。固定 archived raw-timestamp DF seed3407 checkpoint 与 Day90 source-statistics Tent，102 个 directed oracle 网站组中 self accuracy 平均 +1.297 pp，但 cross 仅 -0.250 pp、负向 40.2%，mixed 仍 +0.578 pp，cross-damage 与 mixed-cancellation gate 失败；随机对照和幅度审计通过也不能挽救。第二关未执行，未生成 selector/signal/Burst 方法。该结果是 TemporalDrift development evidence，不替 Host 作更大路线决定。WTT-Time/AWF 保持封闭。

已完成：共享数据迁移；独立项目组织；旧研究交接；统一实验记录入口；2026-09-15 更新 PROTOCOL.md v1，确定 TemporalDrift 开发机制、WTT-Time/AWF 评价冻结方法的数据角色与信息权限。

基础模型：已最小迁入 DF、原双分支 VarCNN，并新增方向单分支 VarCNNDirection，位于 src/ta_wf_next/models。来源与接口见该目录 SOURCES.md。迁移时 4 项 CPU 合成输入检查通过。当前 screening 数据适配器与训练评价入口已实现，真实 source 仅用于隔离/长度审计和随机初始化 smoke，尚无训练。显式使用旧环境中的 PyTorch，新包没有旧代码运行时依赖；新项目独立依赖环境尚未安装。

下一项：2026-09-22 第二轮审计 `20260921T185436Z_proteus_identity_isolation_e596ccf3` 完成：23文件475683行视图记录完整内容审计；Version恢复90145条唯一版本锚点，四份drift共有10017条未知身份，048目标可恢复64406条045/046/047样本（各102类）。Network/Behavior除共享源文件外，完整trace及前5000方向无额外交叉；subpage有2661 URL重复副本。抽样736行中459行绝对时间回退，时间token需先核实语义。未训练评分、未建split。建议下一步冻结开发/留出角色并做时间/有效长度审计；共享结构生成器仍为候选。外部WTT/AWF未开放，既往共享干扰阴性结论不变。首轮结构审计与本轮详细证据均保留。

最近完成：`20260921T191053Z_burst_tokenizer_prototype_ef77e595`，共享方向run生成器v0位于src/ta_wf_next/burst_tokens.py，精确与coarse-only视图分离；边界/padding/预算规则见BURST_TOKENS.md。12项新测试通过；全项目21通过1跳过；三场景384条真实source/subpage样本接口检查0错误。未读标签/训练评分，未实现模型adapter或时间分支，稳定性/泛化未验证。下一步冻结开发/留出及表示对照后才考虑训练。此前时间审计结论保留：方向内单调，跨方向gap不可信，采集根因缺转换链；不排序修复。外部WTT/AWF未打开。

旧研究：R3 共享修正候选 STOP；R3-E 删除 donor 位移后保留主要收益。详见 HANDOFF.md。
