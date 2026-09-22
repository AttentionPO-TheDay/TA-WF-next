# Execution record

执行日期：2026-09-18（Asia/Shanghai）。工作目录固定为 `/home/rbf/TA-WF-next`。

## Authority and immutable inputs

完整读取并核对 `AGENTS.md`、`STATUS.md`、`PROTOCOL.md`、`HANDOFF.md`、`configs/datasets.json`，上一阶段 `exp_bdca1fca83844035` 的 plan/results/position/boundary/summary/code，以及正式 v3 source manifest、`exp_9121b664a1854097` 的 protocol/checkpoint/prediction artifacts。正式 split SHA-256 为 `0f322e9a…f3f142`；DF checkpoint SHA-256 为 `1bf85127…9133ae`。数据、split、checkpoint、正式预测和前序实验均只读。

## Plan freeze and source-only preparation

- 先写 `CORRECTION_PLAN_v1.md`，再运行 source-only `prepare`。
- v1 prepare 在打开 future 文件前按规则停止：v3 supervised-train 的 class 4 缺少 common-500 eligible trace。随后 source-only audit 确认 roles union 对 class 4/8/23/43 都无 eligible trace，official valid 也不能补齐。
- Host 授权后写 `CORRECTION_PLAN_v2.md`，冻结 common-500 不变、四类 exact-zero additive local contribution、102-class strata 与 global-unsupported bypass sensitivity。plan SHA-256 `15a5db0d…34e1320` 写入 `plan_freeze_v2.json`；当时 Day0 参数和 future logits 均不存在。
- `/home/rbf/TA-WF/.venv/bin/python .../run_correction_diagnostic.py prepare`：使用 train/valid 与冻结 checkpoint；主 alpha 及两个邻近 run alpha 均为 0，唯一非零候选是 main-window outgoing fraction alpha 0.1。Day0 parameter SHA-256 `fc2513c7…720d18`。

## Implementation correction and future inference

- 首次 future inference 错用 batch 256，在强制 cache 对齐处停止；Day14 22,603 行中只有两个 near-tie argmax 与 formal cache 不同，最大 competing-logit gap `9.48e-4`。未生成诊断 CSV 或裁决。失败 logits 与第一次 Day0 freeze 以 `attempt1_batch256_*` 保留。
- 修正为正式 `run_temporal_screening.py` 的 batch 128 后重跑 source-only prepare；所有 Day0 alpha 选择完全不变。该修复只恢复数值重现，不改数据、模型、feature、scale、grid 或 gate。
- batch-128 `score` 对 Day14/30/90/150/270 的 source file、row IDs、truth scoring array 和 formal A argmax 全部 exact match。新 logits 只写入本 run。GPU forward wall time分别约 1.109/0.896/1.130/0.933/0.781 秒；没有 optimizer、backward、参数更新或 adaptation。

## Checks

- `python -m py_compile .../run_correction_diagnostic.py`: pass.
- `run_correction_diagnostic.py verify`: 10 required result files present; 72 fusion rows；每行 `Net Repair = wrong→correct - correct→wrong` 且 accuracy delta arithmetic pass。
- `python3 scripts/experiment.py check`: 12 experiments and all configured dataset roots pass；不加载 arrays/labels/checkpoints。
- `git diff --check`: pass.
- Required CSV row counts including header: local rank 43, error complementarity 10,392, fusion 73, specificity 49, per-site/date 511.

## Resource and permission record

Training runs 0；adaptation runs 0；new checkpoint 0。只运行既有 frozen DF checkpoint inference。没有 Burst CNN、局部网络、selector、learned gate、TTA 或后续 Job。future labels only entered after the Day0 support set, templates, scales, alphas and predictions were fixed, for development metrics and attribution.
