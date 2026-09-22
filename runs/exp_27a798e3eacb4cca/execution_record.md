# Execution record

- 工作目录：`/home/rbf/TA-WF-next`
- 日期：2026-09-17（Asia/Shanghai）
- 操作：读取协议、数据目录清单、source split 代码/manifest、正式 schema/provenance 与隔离审计、指定前序 experiment 记录；用 `rg`、`find`、`sed`、`jq`、`sha256sum` 做只读核对。
- 未执行：训练、微调、backbone 解冻、GPU、模型推理、embedding 提取、数据/checkpoint/前序 artifact 修改。
- 协议偏差：一次使用旧环境 NumPy 的全目录 schema 检查写成了对每个 NPZ 成员求 shape/dtype，因而触发数组加载；命令在 30 秒后超时且返回空输出。无法判断超时前操作系统实际读到了哪些 future 成员，故 future 物理访问记为 `uncertain`，而非 0。没有 future 内容被本审计接收、保存、分析或用于选择条件/阈值/结论；随后所有 schema/计数均改为引用既有正式 audit。
- 写入范围：仅 `runs/exp_27a798e3eacb4cca/` 及项目 registry/status 的完成记录。
- 关键输入哈希：见 `summary.json.input_sha256`。
- 未解决风险：官方 release 缺少 row-level acquisition/session/source-family manifest；精确哈希不能替代该 provenance。若未来取得官方 manifest，需新建版本化审计，不能回写本 v1 裁决。
