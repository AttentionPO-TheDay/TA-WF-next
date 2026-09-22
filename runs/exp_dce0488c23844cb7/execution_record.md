# Execution record

## Actions

1. Read `AGENTS.md`, `STATUS.md`, `PROTOCOL.md`, `HANDOFF.md`, dataset registry, current formal split/checkpoint records, and completed TemporalDrift/TTA/batch-composition evidence.
2. Audited archived DF source checkpoint, raw timestamp pipeline and Day90 Tent implementation. Froze `INTERFERENCE_DIAGNOSTIC_PLAN_v1.md` and `config.json` before producing this Job's evaluation results.
3. Added a self-contained minimal archived DF/Tent implementation under this run only. It does not import the old project at runtime.
4. Ran four CPU structural tests: checkpoint/scope, transition metrics, label-free adaptation signature and Spearman implementation. All passed.
5. Ran the complete GPU matrix twice with identical frozen settings. The first complete run established the result; the second was a schema-only correction adding explicit No update rows and per-row magnitude fields. No scientific setting changed. Each run performed 51×3 oracle plus 5×51×3 random temporary TTA updates = 918; total 1,836.
6. Ran a separate full-Day90 static inference and exactly reproduced archived accuracy, providing pipeline compatibility evidence.
7. Per the failed first gate, did not compute second-stage signals and did not create the three conditional signal CSVs.

## Commands/tests

- `/home/rbf/TA-WF/.venv/bin/python` direct calls for four test functions and `py_compile`.
- `/home/rbf/TA-WF/.venv/bin/python scripts/experiment.py check` → 13 registered experiments and all dataset roots OK before completion update.
- `nvidia-smi --query-gpu=...` → GPU 0/1 idle; execution used CUDA.
- `/home/rbf/TA-WF/.venv/bin/python runs/exp_dce0488c23844cb7/code/run_interference_diagnostic.py --device cuda` (two complete, identical-setting executions).
- Independent CSV/JSON row-count, transition-count, gate and artifact-presence checks; `git diff --check`.

## Outputs and boundaries

All new files are confined to this run directory except the required `EXPERIMENTS.csv` and `STATUS.md` registry/status updates. Dataset, checkpoint, formal splits and prior artifacts were read-only. No WTT-Time, AWF, future external test, model training, new checkpoint, hyperparameter search, Burst feature or grouping algorithm was used.

Unresolved risks: the archived raw-timestamp DF lineage differs from the newer sign-only formal checkpoint lineage, so this result must stay attached to the audited archived combination; one checkpoint/date does not estimate training-seed/date variance; random controls do not create independent checkpoint repeats; literature search is bounded rather than exhaustive.

Final artifact SHA-256: matrix `5fe1f9bee623e5e8f250ee2158798f2089673be918ba8580ef4ca1627104b7ac`; pair table `d964ad999eb8907afac53961050abb323039ca9a86c60e0f9060ce4c092f3555`; random controls `d77b94074fb94fc6f62225f60bfd0788ee0ec699df1ca48beb0592207c772dc1`; magnitude audit `975e5bb529a7087cfd160f934a2aabc5fc30adebd1c7c2cdf874cdf521c177ed`; aggregate table `133aa8b135d355a68ba4fae696267041b91bed33e996cca1b97f821cfd2edc70`; summary `b0605164bb68a86bec8d8dc83ef7611c467b5a50ec26118cab579d85df45197c`.
