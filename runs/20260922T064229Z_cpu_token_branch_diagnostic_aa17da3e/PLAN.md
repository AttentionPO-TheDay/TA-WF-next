# 20260922T064229Z_cpu_token_branch_diagnostic_aa17da3e

问题：固定hybrid模型中连续长度分支的尺度与full/off/bias_only推理干预是否提示融合干扰？

状态：frozen，运行前冻结。

只读复用本项目 typed_encoding run 的 hybrid 三 seed 已选 checkpoint、source2040/valid510 缓存和历史预测；显式 importlib 导入其 train.py 中 TokenNet，不依赖旧项目代码。通过既存 seal 验证全部输入。source/valid 标签仅用于开发诊断评分，不训练、不选模、不调干预强度。未来、WTT、AWF、适应关闭。

预定三种推理条件：full 原样；off 关闭 continuous 层整个输出（含 bias）；bias_only 保留 bias、删除输入相关 weight*x。不根据结果增加干预或选择最佳系数。full valid 须逐条复现历史预测。输出各条件 accuracy/macro-F1/预测变化率/零召回类数；分支尺度仅统计非 padding token 的 RMS，分别记录 bucket、continuous、continuous bias、variable component、direction、position。

seeds 1729/3407/2026，CPU 单线程，CUDA_VISIBLE_DEVICES 为空，timeout 180 秒。无训练预算；每角色每 seed 3 次前向。失败即停止，不重复训练。封存代码/配置/输入/预测，独立重算全部指标并核对 full 复现。干预属训练后分布外操作，不能证明连续信息在重新训练时无用，不能证明抗漂移或训练期蒸馏有效。
