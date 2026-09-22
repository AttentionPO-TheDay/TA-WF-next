# Historical Variation Episode Feasibility Audit — Results

完成时间：2026-09-17（Asia/Shanghai）。技术裁决：**FAIL**。

现有历史 source 不能有依据地构造同网站、跨真实变化条件、无泄漏的 3-shot support→query episode。发布的 `train.npz`/`valid.npz` 只有 `X/y`；`X` 是单条 trace 内的有符号相对包时间，不是采集日期。没有 row-level 日期、session、capture/run、客户端/采集器、网络环境、原始文件批次或派生父样本 ID。Day0 是唯一可验证的历史采集条件，所以至少两个真实条件的网站为 **0/102（0%）**。

样本量不是瓶颈：train 每站 163–198，valid 每站 18–22；但 `train/valid` 只被正式协议证明为训练/选模 split，未被证明是两个真实采集条件。将其、随机 source split、行号、trace 长度或随机删包/裁剪/扰动当作变化轴均会伪造任务。历史 source 内也没有可声称的 earlier→later 方向；最多只能构造普通 Day0 i.i.d. split。

精确去重证据通过但不能改变裁决：v3 排除 168 个 train→valid admitted-input overlap 和 718 个 train 内重复，保留 18,553 个 canonical train rows；v4 对 raw/admitted representation 均报告零 cross-role exact duplicate。valid 仍为 2,160 rows / 2,151 unique content groups。由于缺少 session 和 derivation-family ID，这些 hash 不能排除同 session 重采、切片、近重复或其他同源关系，故无法给出 episode-level source independence 保证。

按本 Job 的规则，`CONDITIONAL` 也不成立：这里不是“真实轴存在但覆盖稀疏”，而是历史 source 中第二个可验证条件及其分组元数据不存在。论文证明 Day0 与 later-day corpus 是真实时间变化，但本门槛禁止用 TemporalDrift future/query 标签定义历史 episode；未来日期的存在不能补成 source-side 训练轴。

因此不提供训练草案，也不创建训练 Job。建议停止“用当前 historical source 构造真实变化 episode 来学习表示”的当前版本，除非取得可信的 source-side capture/session/date/environment manifest 和可传播到所有派生样本的原始 trace ID；不得用随机 split 或增强替代。该 gate 裁决不替 Host 作更广泛的研究路线决定。

执行完整性偏差：一次只读全目录 NPZ schema 命令误触数组加载并在 30 秒后超时，无任何输出返回。因此不能声称 future 文件零物理访问；可以确认的是，审计没有取得或使用任何 future 数值/标签/统计，裁决仅依赖 source-side schema 和既有正式 provenance/isolation artifact。没有据此调整轴、阈值或结论。

审计详情：`AUDIT_PLAN_v1.md`、`metadata_source_audit.md`、`candidate_axis_stats.{md,csv}`、`episode_feasibility.md`、`leakage_audit.md`；机器摘要：`summary.json`。
