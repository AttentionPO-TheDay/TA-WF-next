# 实现修复与报告补齐

1. 初始版本 20261014T050956Z_tent_baseline_2805cba7：预检2步support smoke创建events.jsonl，正式runner误认为重复执行，未运行正式任务。
2. v2 20261014T065552Z_tent_baseline_v2_87cc8508：允许预检事件，采用launch.json与runner_started.json防重复；首个800步TENT任务完成后，历史训练流断言失败。source loader seed错误使用2026，应为2025；target仍为2026。本次未产生query评分，失败产物保留。
3. v3：修正source loader seed至2025。原TENT算法、BN参数作用域、support/query划分、800步预算、学习率与query规则保持一致；18项正式任务及独立审计通过。
4. 正式runner遗漏report()调用，完成登记时RESULTS仍为模板。基于已审计scores补齐RESULTS、verdict、paired contrasts，并修订STATUS及登记摘要；未重跑或修改冻结代码。
