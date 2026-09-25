# 20260922T015431Z_cpu_tiny_multiview_learning_2a2e1036

问题：在统一抽样与固定 CPU 训练预算下，packet、window、exact-run、coarse-run 生成器输出是否提供可学习的网站识别信号。

状态：frozen_cpu_learning_probe。Proteus NetworkDrift/train 每类 20 条作监督训练，valid 每类 5 条只用于每 epoch 选 checkpoint；JP 每类最多 20 条、BehaviorDrift/subpage 每类 20 条只作已暴露 TemporalDrift 开发评分。抽样 seed=1729，与前两次 CPU probe 相同。WTT/AWF 不读取。

对照与模型：四个独立轻量模型，packet 为小型 Conv1d，window 为两层 MLP，exact/coarse run 为共享形式的 token projection + masked mean/max pooling + 分类头。输入 observation budget=5000；run token width=512，显式 mask；window 保持 50/250 两组比例/转向率。每个模型参数和 source 标准化统计独立估计。

训练：CPU only，seed=1729，AdamW(lr=1e-3, weight_decay=1e-4)，15 epochs，batch=128；以 source valid macro-F1 选择最佳 epoch，不查看 JP/subpage 后返调。预测先封存再评分。无 selector、融合、TTA 或 checkpoint 复用。

停止条件：数据结构、类别完整性、非有限值或输入维度异常即停止。CPU 4线程，总运行墙钟上限20分钟，单seed四模型共60 epochs；预算超限保留已有产物，报告未完成项。

具体网络：packet Conv1d(1,16,k9,s4)-ReLU-Conv1d(16,32,k7,s4)-ReLU-AdaptiveAvgPool(32)-Linear(102)；windows Linear(240,128)-ReLU-Linear(128,102)。run 为逐token Linear(C,64)-ReLU、masked mean/max、Linear(128,128)-ReLU-Linear(128,102)，无位置编码，不能验证有序run编码器的上限。exact通道(direction,log1p count,左右观察边界)，coarse通道(direction,log2 bin,前后bin差及有效位,左右边界)。截断后最后token的next bin允许引用5000观测内的下一run，显式不是严格512-token信息预算。coarse不传精确包数。

packet/window按source逐位置标准化；run按source有效token逐通道标准化，padding重新置零。15个checkpoint选择机会；不同架构参数量、压缩和截断不同，本实验不声称等容量公平比较。JP/subpage是网络/行为条件，不是时间日期，不能据此证明抗时间漂移。结果仅回答轻量可学习读出的开发效用。
