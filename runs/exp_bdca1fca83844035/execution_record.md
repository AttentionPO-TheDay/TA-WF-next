# Execution record

执行日期：2026-09-18（Asia/Shanghai）。工作目录固定为 `/home/rbf/TA-WF-next`。

## Read-only authorities checked

- `AGENTS.md`, `STATUS.md`, `PROTOCOL.md`, `HANDOFF.md`, `configs/datasets.json`.
- Formal TemporalDrift loader/transform in `scripts/run_temporal_screening.py` and `src/ta_wf_next/screening.py`.
- Formal v3 manifest and audits under `runs/exp_6238dacf9aa142cc/artifacts/`.
- NPZ member headers: uncompressed `float64 X[N,10000]`, `float64 y[N]`; exact row counts recorded in the plan.
- Full SHA-256 verification of the six used NPZ files and v3 split matched the formal manifest.

## Commands and checks

- `/home/rbf/TA-WF/.venv/bin/python .../run_burst_diagnostic.py --self-test`: passed exact synthetic signed-RLE checks.
- `python3 scripts/experiment.py check`: passed; all registered experiments had plans and dataset roots existed.
- `run_burst_diagnostic.py`: first run completed all raw scans and primary outputs, then stopped in `equal500` sensitivity because it assumed every site had an eligible Day0 trace. No input or prior artifact was written.
- The sensitivity aggregator was corrected to use the explicit intersection of covered sites and report coverage. `run_burst_diagnostic.py --resume`: completed; the second scan reproduced every first-pass date count exactly and reused the already complete primary position table.
- `run_position_sensitivity.py`: completed fixed-500 two-coordinate sensitivity on 96 common sites; rerun after adding terminal-run exclusion produced both retained/excluded scenarios.
- `summarize_outputs.py`: generated reports and machine-readable summary from frozen CSV/JSON results.
- `sha256sum` verified train/day14/day30/day90/day150/day270 and formal split; hashes appear in the audit.
- Final checks include rerunning both script self-tests/summary rendering, CSV/JSON structural validation, `scripts/experiment.py check`, `git diff --check`, and confirmation that no training/checkpoint/GPU process or output was created.

## Resource boundary

The diagnostic used one CPU process and memory-mapped immutable NPZ members. Peak RSS of the completed main run was about 5.0 GB; elapsed time was 13m55s. Training runs 0, adaptation runs 0, GPU use none. No dataset, checkpoint, split, previous run, or formal result was altered.
