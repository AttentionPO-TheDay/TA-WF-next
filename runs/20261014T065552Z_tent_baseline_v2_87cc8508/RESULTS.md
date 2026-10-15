# 实验结果：执行失败

首个800步support适应完成，但source loader seed未对齐历史基线导致训练流断言失败；未产生新query性能结果。

修订原因：原RESULTS为“尚未执行”模板，与failure.json不一致；据日志纠正状态。失败产物保留，修复成功版本为 20261014T070454Z_tent_baseline_v3_83d48bf3。
