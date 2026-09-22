# Execution record

日期：2026-09-16（Asia/Shanghai）。工作目录始终为 `/home/rbf/TA-WF-next`。

- 训练前完整读取项目协议、HANDOFF 和四个指定旧实验的计划、结果、metadata、manifest、common-query、selection diagnostics、integrity 与 execution artifacts；完整解析 20 MB support manifest 和 3.4 MB common-query manifest。
- 在任何新增训练前冻结 `REPLICATION_PLAN.md`，SHA-256 `e3745039e54eed4e531c839671a2daf2c288fd3fe7d3aab71ffbb77c6f638614`；seeds 一次性固定为 1013/2024。preflight 校验 donor/data/split/rule/evaluation hashes，0 errors。
- 验证：`py_compile`；4 个 implementation unit tests；正式 preflight；4 次固定 GPU training；4 次 source-only G-source/prototype preparation；12 个 architecture×training-seed×date evaluation；aggregate；独立 verifier。
- 新训练严格为 4，checkpoint 严格为 4。原 seed 6238 checkpoint/result 未重训、未改写。新增 evaluation 72 个，旧正式 evaluation 36 个只读复用。
- 一次 frozen evaluation 在写出 artifact 前因外部 GPU 占用 OOM，按同配置改 CPU 重跑；两次未写出当前 date artifact 的评价为解除 BLAS 争用而中止并同配置恢复。没有替换 seed、候选、solver、迭代或指标定义。
- WTT-Time、AWF、其他封闭最终评价、adapter、attention、meta-learning、特殊 loss、无标签 TTA、外部数据、多日期记忆均未使用。
