# Global prediction audit

- Checkpoint: `/home/rbf/TA-WF-next/runs/exp_9121b664a1854097/checkpoints/df_best.pt`
- SHA-256: `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae`; DF seed 6238, epoch 29.
- Split: `/home/rbf/TA-WF-next/runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`; SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`.
- Input: sign of raw positions 0..4999. Frozen eval/inference only; no optimizer, backward, parameter update, adaptation or batch-label access.
- Existing formal cache stores only top-1; fresh float32 logits were materialized in this run for rank/fusion. Every date matched cache source file, row IDs, truth array and A argmax exactly.

| date | rows | local eligible | global accuracy | device | inference seconds |
|---|---:|---:|---:|---|---:|
| day14 | 22603 | 18234 | 0.712560 | cuda | 1.109 |
| day30 | 22867 | 18493 | 0.650982 | cuda | 0.896 |
| day90 | 28599 | 23048 | 0.565474 | cuda | 1.130 |
| day150 | 24064 | 19869 | 0.505028 | cuda | 0.933 |
| day270 | 19935 | 16393 | 0.460747 | cuda | 0.781 |
