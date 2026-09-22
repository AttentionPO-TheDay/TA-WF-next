# 20260921T184917Z_proteus_condition_audit_f033070f

问题：核实 Proteus 多条件结构及 provenance，评估共享 burst tokenizer；只读结构审计，不训练评分

状态：frozen audit-only，2026-09-22 Asia/Shanghai。Tor 版本纳入范围。

先读 STATUS/PROTOCOL/HANDOFF 和旧六场景审计。数据由 configs/datasets.json 的 data_root 定位，五个额外 Proteus 同级目录由旧官方审计明确定位。只枚举六组原始 NPZ、读取 ZIP directory/NPY header，对比旧证据的 shape、dtype、成员 CRC 和大小。CRC 不代替内容哈希；旧标签统计继承而非重新验证。不读取数组值、标签数组、checkpoint；不打开 WTT-Time/AWF。

单 CPU、无 GPU/训练/适应/评分/选模，无 seed 或数据重划分。差异出现时记录，停止进一步数组读取。产物为 header_audit.json 和 RESULTS.md。Version 跨视图整行匹配、全量去重、数值质量及正式 split 审计仍待后续，不称本次为训练就绪审计。候选生成器仅作设计分析，不实现。
