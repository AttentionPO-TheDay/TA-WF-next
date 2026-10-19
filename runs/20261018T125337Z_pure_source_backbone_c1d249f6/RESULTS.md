# 纯净源域 wide_bn 基石模型结果

完成三 seed 从头训练、valid 预测与 checkpoint 重载。源期指标来自已观察的官方 valid；无独立 Day 0 test。

| Seed | Best step | Valid Accuracy | Valid Macro-F1 |
|---:|---:|---:|---:|
| 42 | 12800 | 88.565% | 88.405% |
| 100 | 12800 | 87.870% | 87.671% |
| 2026 | 10400 | 88.287% | 88.127% |

三 seed 均值±样本标准差：Accuracy 88.241±0.350%，Macro-F1 88.068±0.370%。

输入清单和源文件散列见 artifacts/data_preflight.json；逐 seed checkpoint、history、预测与重载报告见 checkpoints/ 和 artifacts/。未访问未来日期或外部数据。

独立权重复核（`audit_weights.py`，记录见 `artifacts/weight_audit.json`）：三份文件的 SHA256 与训练报告一致，配置散列匹配，89 个状态张量均为有限值且严格加载成功。CPU 重算的 2,160 条 valid 预测、Accuracy、Macro-F1 和 20 次验证历史选出的最早最优步数全部一致；CPU 与原 GPU logits 的最大绝对差为 0.0057–0.0074。使用训练相同的确定性 GPU 设置重算时，三 seed 的 logits 与保存值逐元素完全相同。此复核只验证权重和已保存的源域 valid 结果，不增加独立测试证据。
