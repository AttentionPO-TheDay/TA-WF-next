# 20260922T114658Z_cpu_run_window_adapters_v2_148f23cd

问题：修正首个run/window实现的前向维度错误后，检验两视角互补性及分支专属适配器。

数据固定为既有TemporalDrift source2040/valid510清单，source训练、valid选模/开发评分；不读取未来日期/WTT/AWF，不在线适应。生成器规则冻结；只训练编码器、适配器和分类头。

输入为coarse run（512槽，direction/bin/position/mask）及direction windows（50、250窗口的positive_fraction/transition_fraction，共240数值）。run_pair和window_pair是容量同类双支伪对照；fusion为run+window；fusion_shared共享32→16→32零初始化适配器；fusion_specific为run/window各自32→8→32零初始化适配器。所有分支独立编码后concat分类，参数量实际记录，不作严格等容量结论。

三seed、15epochs、batch128、AdamW、CPU3线程、600秒上限。预注册fusion相对单视角pair至少2/3 seed F1正且均值正才保留候选；specific相对shared若至少2/3正且均值正则说明分支专属处理值得继续。valid不是独立确认。
