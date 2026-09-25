# 20260922T114213Z_cpu_run_window_adapters_faf3c613

问题：容量近似匹配下run/window互补性及分支专属适配器是否有增量

状态：frozen；CPU开发实验。

问题：生成器的run token与direction-window token是否互补，以及各自绑定轻量适配器后能否用于融合。固定生成器规则，不训练字段生成；适配器与分类骨干联合训练。source2040/valid510沿用已封存清单，不重划，source训练、valid选模/开发评分，未来/WTT/AWF关闭。

输入：signed_timestamp前5000方向；run为前512个coarse token（direction embedding、bin embedding、位置编码、mask），window为50/250窗口的positive_fraction与transition_fraction，共240个固定位置特征。窗口不是时间漂移适配器，不使用标签、日期或边界真值。

模型：run_adapter为32维run token残差8维适配器、两层保序卷积、16段masked pooling，输出128维；window_adapter为两层MLP(240→64→128)并有8维瓶颈残差适配器；fusion为两支各自独立处理后concat256→102分类。三臂都从头训练，适配器零初始化确保初始分支近似原表示；run/window单臂用于判断互补性，fusion用于判断联合收益。容量不同，故只作候选筛查，不能称严格公平的最终对照。

预算：3 arms×3 seeds、15 epochs、batch128、AdamW lr0.001 wd0.0001、CPU3线程、600秒上限。valid macro-F1选模、平局早停；全seed逐项报告。预注册fusion相对run_adapter与window_adapter平均accuracy/F1均正，且至少2/3 seed F1正，才继续作为候选；不作漂移、蒸馏或在线更新结论。

校验：生成器重建方向/run/window，source/valid hash隔离；padding mask、适配器梯度与零初始化；checkpoint重载、指标/选模独立重算。保存输入、配置、代码和输出hash。复用旧虚拟环境依赖，不加载旧checkpoint或旧训练代码。此轮仅验证表示互补与适配器可训练性，valid已多次开发。
