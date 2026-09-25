# 实验结果

## 完成

实现了 `src/ta_wf_next/batch_views.py`，提供 packet、exact-run、coarse-run 和 direction-window 的独立 batch adapter。每个输出都显式返回：

- 右侧 padding 的 `values`；
- `mask`；
- `token_count` 与原始 `source_observed_count`；
- 是否超过 token 宽度的 `truncated`。

coarse-run 不含 exact run count，只保留 log2 bin、相邻 bin 差及其有效位和边界标记。time/size 暂不转换。

## 验证

合成测试覆盖 mask、padding、token 截断、partial window、空 batch/非法宽度以及 exact/coarse 分离；与既有 traffic view 测试合计 14 项全部通过。测试使用 `/home/rbf/TA-WF/.venv/bin/python`，不读取真实数据，不读取标签，不训练，不使用 GPU。

这项工作只完成输入适配，不构成 learned baseline 的效用证据。下一步可以在 GPU 空闲前直接进行 collator 与模型接口的 CPU smoke/小样本过拟合测试。
