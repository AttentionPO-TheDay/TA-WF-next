# 20260925T161437Z_transformer_fullsource_capacity_cpu_0de7931a

问题：此前 Transformer 只用510条5-shot标签，不能直接与使用全部source标签的DF比较。本轮将相同固定source2040条标签全部用于自有Transformer监督训练，测量其基本网站分类能力与历史DF/VarCNN水准的差距；不设计或测试漂移模块。

状态：frozen。TemporalDrift source/valid 已参与多轮开发，本轮为已观察开发基准，不是独立确认。

数据与权限：只读使用 `20260925T130044Z_transformer_temporal_pretrain_fewshot_7098625e/artifacts/prepared.pt` 中已固定 source2040/valid510 的 packet/run/window 方向派生 token；SHA-256 `4dc489bf088246c59b73cdb071bb22459c975051ed84b27962bbbf0067d70b24`。source全部2040条标签可进入监督梯度；valid510标签仅用于预定checkpoint选择及开发评分。五未来日期、WTT-Time、AWF、外部PCAP均不读取。没有参考/query适应。

模型：本项目已有 `GeneratorTokenTransformer` 平铺多视角（110400参数）与 CipherSight启发的 `HierarchicalViewTransformer` 视角内/跨视角层次化编码（113670参数），均从头训练，不加载既有checkpoint，不使用无标签预训练、蒸馏、LoRA、adapter或TTA。两者同一packet每50包摘要、前128个run、50/250 window；不改输入或标签权限。

训练及选模：两条件×seed1729/3407/2026，每组100 epochs，batch64，AdamW lr0.001/weight_decay0.0001，CPU每条件最多4线程；每5轮对source2040和valid510以eval mode计算accuracy/Macro-F1，按valid Macro-F1在5,10,…,100选最早最佳。两条件全部结果和所有seed报告，不事后挑seed；最长wall-time每条件5400秒，两个条件并行启动。非有限loss、输入hash不符、超时或资源异常立即停止并保留失败/部分记录。

比较与解释：主结果为valid accuracy和Macro-F1、source拟合、最佳epoch、预测类别覆盖、与历史DF-only 49.150% accuracy/48.144% Macro-F1的绝对差距。历史DF来自 `20260923T104034Z_gpu_df_token_logit_residual_5a92bbaa` 同source2040/valid510方向输入，但45 epochs/GPU、架构和选模机会不同，属于性能标尺而非本轮重新训练的等计算对照。RF没有本工作区同一固定协议结果，本轮不编造其数值。若Transformer距离仍很大，先改输入保序和主干能力；只有达到用户认可的相近水准后再讨论生成器预训练或漂移模块。本轮不根据valid反馈临时改结构、轮数或学习率。
