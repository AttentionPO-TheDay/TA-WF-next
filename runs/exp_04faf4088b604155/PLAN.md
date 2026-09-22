# 3-shot Estimation Noise vs Historical-Mode Coverage Diagnostic

Experiment: `exp_04faf4088b604155`。问题：在 `exp_b471517a3e6f41e7` 的冻结最终 global embedding 与正式嵌套 support/query 上，3-shot 相对 10-shot 的损失更像有限样本估计噪声，还是遗漏了 source 历史已经支持的网站内表示区域？本实验只做可观测失效诊断，不提出或搜索新网络/适配器。

状态：running。正式、不可回写的诊断定义见 `DIAGNOSTIC_PLAN_v1.md`；机器输入哈希由只读 `preflight` 写入 `artifacts/input_manifest_v1.json` 后才允许提取 embedding。新增 backbone 训练/微调预算为 0。

