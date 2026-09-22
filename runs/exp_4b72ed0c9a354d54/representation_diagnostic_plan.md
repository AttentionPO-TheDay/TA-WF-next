# Bounded representation diagnostic pre-registration

## Frozen provenance and permissions

The only backbones are the two source-trained best checkpoints from `exp_9121b664a1854097`:

| Backbone | Checkpoint | SHA-256 | Saved/best epoch | Model definition |
|---|---|---|---:|---|
| DF | `runs/exp_9121b664a1854097/checkpoints/df_best.pt` | `1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae` | 29 | `src/ta_wf_next/models/df.py::DF`, direction sign input `[B,1,5000]` |
| VarCNNDirection | `runs/exp_9121b664a1854097/checkpoints/varcnn_direction_best.pt` | `fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83` | 23 | `src/ta_wf_next/models/varcnn.py::VarCNNDirection`, explicit direction-only derivative, not full VarCNN |

Checkpoint metadata and the 30-epoch histories agree on those best epochs. Both came from seed 6238, AdamW lr 0.001/weight decay 0.0001, and maximum official source-validation macro-F1 with earliest exact tie. No backbone optimizer or training entry point is permitted in this experiment.

The read-only v3 split is `runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json`, SHA-256 `0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`: 16,309 supervised-train, 204 reference (2/class), and 2,040 source-holdout (20/class), mutually disjoint; official `valid.npz` remains source validation. Its seven recorded NPZ hashes and v4 content audit must pass. Linear models fit only supervised-train. Validation selects regularization/configuration; reference and source-holdout never fit or select. Future inputs never fit, select, update references, or choose layers/aggregation/readout. Future labels enter only after all predictions are fixed.

## One shallow and one preselected intermediate layer

The intermediate layer was selected once from architecture depth, receptive field and interpretable coverage before any new future representation was extracted. It is the end of the convolutional stack's first half for each architecture; there is no second candidate intermediate layer.

| Backbone/tier | Hook layer | Shape for length 5000 | stride | effective RF and inclusive bounds | structural full-RF positions | global / ordered-4 dimensions |
|---|---|---:|---:|---|---|---:|
| DF shallow | `feature_extraction.0` | `[B,32,1249]` | 4 | 22, `[4i-8,4i+13]` | `i=2..1246` | 32 / 128 |
| DF intermediate | `feature_extraction.1` | `[B,64,311]` | 16 | 106, `[16i-40,16i+65]` | `i=3..308` | 64 / 256 |
| VarCNNDirection shallow | `dir_encoder.convs.0` | `[B,64,1250]` | 4 | 35, `[4i-17,4i+17]` | `i=5..1245` | 64 / 256 |
| VarCNNDirection intermediate | `dir_encoder.convs.3` | `[B,128,625]` | 8 | 363, `[8i-181,8i+181]` | `i=23..602` | 128 / 512 |

DF uses even-kernel same-padding convolutions followed by unpadded max pooling; VarCNNDirection uses constant-zero boundary padding in the initial convolution, max-pool padding, and dilated residual blocks. A position is effective only when its complete theoretical RF is inside input indices 0..4999 and every covered admitted input is finite and nonzero. Zero remains “unknown/padding”, including interior zero. We do not infer length from counts, retain padded positions, or filter short samples. Global mean uses all effective positions; ordered-4 splits the effective positions in sequence into four near-equal contiguous chunks, averages each chunk, concatenates in order, then L2-normalizes the whole vector. Samples with no effective global position or fewer than four ordered positions remain in evaluation/fitting as an all-zero representation and are reported, never filtered.

The exact source-only audit is `artifacts/layer_source_audit.json`. Observed-span min/median/max are 109/1037/5000 for supervised-train, 153/1002/5000 for reference, 128/1024/5000 for validation, and 129/1041.5/5000 for holdout; respectively 15,501/16,309, 195/204, 2,065/2,160 and 1,931/2,040 are shorter than 5000. No source trace is all-zero.

| Backbone/tier | supervised-train `<4` / zero | reference `<4` / zero / eligible classes | validation `<4` / zero | holdout `<4` / zero | effective-position median (train/ref/val/holdout) |
|---|---:|---:|---:|---:|---:|
| DF shallow | 0 / 0 | 0 / 0 / 102 | 0 / 0 | 0 / 0 | 254 / 246 / 251 / 255.5 |
| DF intermediate | 166 / 1 | 2 / 0 / 101 | 26 / 0 | 15 / 0 | 58 / 56 / 57 / 58.5 |
| VarCNNDirection shallow | 0 / 0 | 0 / 0 / 102 | 0 / 0 | 0 / 0 | 250 / 242 / 247 / 251.5 |
| VarCNNDirection intermediate | 2,007 / 1,728 | 25 / 21 / 92 | 274 / 243 | 248 / 214 | 84 / 80.5 / 83 / 85 |

This is a pre-future structural limitation, not grounds to change the unique intermediate layer. Every short trace stays in fitting/scoring. In particular, all-zero intermediate vectors remain present, and cosine retains all 204 references even where a reference representation is zero; no class or row is silently removed.

## Fixed diagnostic matrix and readouts

For each backbone, exactly four vector representations are evaluated: shallow/intermediate × global-mean/ordered-4-concat. Each has:

- cosine 1-NN against the same 204 frozen reference traces (two/class), using L2-normalized representations and earliest reference index on an exact score tie;
- L2-regularized multinomial logistic regression fit on all and only 16,309 supervised-train representations. `solver=lbfgs`, `C in {0.1,1,10}`, `max_iter=300`, `tol=1e-4`, intercept on, no class weights. Source validation macro-F1 selects C; exact ties take smaller C. No penalty/solver/scaler/feature/layer/region search is allowed.

At each of the two layers, the existing order-insensitive regional rule is also retained: normalize each of four region means; for every query region take the best cosine among a reference's four regions, average those four maxima, then choose the best of 204 references. It has no linear analogue because it is a pairwise matching rule, not a fixed vector. Thus the bounded matrix has exactly 10 rows/backbone: 2 layers × (global cosine, ordered cosine, global linear, ordered linear, unordered regional cosine).

The linear model has roughly 160 labeled examples/class, vastly more supervision than cosine 1-NN's two references/class. It is a representation diagnostic, not a fair or deployable local recognition baseline; improvements cannot be attributed to the two-reference local mechanism.

Old frozen A/C/D/E source-holdout results from `exp_9121b664a1854097` are copied as provenance-marked controls. C is the hard full 512-d embedding/two-reference global 1-NN control. D is the shallow four-region order-insensitive max-region matching control; E is shallow valid-position global mean 1-NN. They are not new repeats.

## Metrics, thresholds and future freeze

Report source validation selection records and source-holdout accuracy, macro precision/recall/F1, 102 per-website accuracies, representation bytes, extraction/readout wall time, iteration/convergence records, and effective-position coverage.

All “clearly/significantly better” diagnoses are deterministic effect-size flags, not inferential significance claims: ≥5.0 percentage points macro-F1 on source-holdout for layer or ordered-vs-global/unordered comparisons; ≥10.0 pp for linear-vs-cosine because supervision differs substantially. A representation/readout is source-usable only if its source-validation macro-F1 is ≥40%. Source-holdout never gates or selects future configurations.

After the complete source matrix, `freeze` selects at most one candidate per backbone among source-usable rows by highest source-validation macro-F1. Exact ties prefer global over ordered over unordered, shallow over intermediate, cosine over linear, then smaller C. If no row reaches 40%, that backbone has no future diagnostic candidate. The selected estimator and representation are hashed into `artifacts/frozen_selection.json`; future execution refuses to run without that artifact and the corresponding source artifact hashes.

Only those zero-to-two selected configs are evaluated on Day14/30/90/150/270. No future result can alter them. Future reports absolute accuracy/macro-F1, drop from the selected config's source-holdout result, old C absolute performance, and selected-minus-C, including per-site results. Repeated positive future benefit means selected-minus-C macro-F1 >0 on at least 4/5 dates for each of both backbones; this is descriptive evidence, not a Host research decision.

## Interpretation and stopping boundaries

- Shallow linear still poor while the matched intermediate linear is ≥5 pp better: evidence of insufficient current shallow representation.
- Ordered concat ≥5 pp above matched shallow global or old unordered D: order retention merits a separate limited test, but is not itself a drift mechanism.
- Source recovery without repeated positive future advantage over C: general recognition recovery only; no support for local drift resistance.
- Only usable source discrimination and repeated positive future benefits in both backbones meet the stated evidence condition for considering later DNNF/TFAN comparisons.
- No clear positive signal ends this local candidate. No extra layer, attention, loss, region count, external dataset, DNNF/TFAN, large hyperparameter search, or backbone retraining may be added here.
