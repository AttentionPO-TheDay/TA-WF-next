# Configuration audit

## Frozen lineage

The executed combination is internally coherent and deliberately does not use the newer sign-only checkpoint.

- Dataset: TemporalDrift `day90.npz`, SHA-256 `eaa25c23657f22509ab3ac9018a63a12736fab007db45cfa2ea05cbb6c0b25a6`, 28,599 rows, 102 labels, 239--293 rows/site.
- Checkpoint: archived DF source-only seed 3407 epoch 30, SHA-256 `27491ed4f488f517cea31a5b50f661e9d90f843ae5a43d0509688de19e6913f4`; 102 classes; source-valid macro-F1 0.8083233769.
- Input: raw signed relative timestamps, first 5,000, float32 `[B,1,5000]`. No sign conversion and no TAM conversion.
- TTA: the archived Day90 Source+Tent setting: all 20 BN affine tensors, source running statistics, model eval/Dropout off, lr 0.005, five entropy steps, max-softmax threshold 0.5, episodic reset, accept only non-increasing support entropy.
- Isolation: site-wise PCG64 seed 20260919; first 64 shuffled rows/site are adaptation-only, all remaining rows/site evaluation-only. There is no row overlap. All 51 fixed adjacent-label pairs cover all 102 sites.

## Compatibility evidence

The minimal local DF definition loads the archived state dict with `strict=True`. A full 28,599-row static Day90 check reproduced archived accuracy exactly (`0.6404419734955767`) and reproduced macro-F1 to numerical rounding (`0.6166884827900728` local versus archived `0.6166885415537683`). The first-64 static-logit fingerprint is `220c1647cf2db86c2314935ac45e73663dbfceae2f75849eb627c15bf96d75f0`.

The current new-workspace checkpoint `runs/exp_9121b664a1854097/checkpoints/df_best.pt` uses `sign(X[:5000])`; it was audited but not loaded. This prevents a historical raw-input checkpoint/TTA rule from being silently mixed with the new sign-only pipeline.

## Label and state audit

The adaptation function accepts only model, parameter names, source parameters and support features; it has no target/label argument. Confidence selection and entropy fallback use support logits only. Evaluation labels enter only `metrics()` after updated logits exist. Every A-only, B-only and mixed update uses 64 support rows and starts from the same immutable source parameter mapping. BN buffers remain at source values.

All 918 updates in the final complete run were accepted. Median effective fractions for oracle A/B/mixed were 0.8875/0.884375/0.85625; median relative parameter deltas were 0.00063825/0.00061682/0.00051743. Their maximum/minimum ratio was 1.234. Contrast correlations with effective fraction and update magnitude were 0.026 and 0.273, below the frozen 0.5 exclusion threshold.

No new checkpoint, source training, backbone modification, external test access, WTT-Time/AWF access, Burst feature or selector was used.
