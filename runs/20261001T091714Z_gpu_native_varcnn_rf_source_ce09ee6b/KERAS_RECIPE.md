# Keras 2.0.8 Var-CNN training recipe audit

Public Keras tag `2.0.8` snapshots were retrieved on 2026-10-01. Source URLs and exact downloaded SHA-256 values are in `artifacts/keras/manifest.json`. These are tag-pinned source snapshots, not a cross-framework numerical-parity claim. No model, checkpoint, or dataset was loaded by this audit.

## Callbacks

Official Var-CNN `var_cnn.py:291–302` installs callbacks in order: ReduceLROnPlateau, EarlyStopping, ModelCheckpoint. All monitor validation accuracy in [0,1], auto mode resolves to max.

- ReduceLROnPlateau: factor=sqrt(0.1), patience=5, cooldown=0, min_lr=1e-5. Keras default epsilon=1e-4; improvement is strictly `current > best + 1e-4`. Best starts at negative infinity. On a failed improvement, **check `wait >= patience` before incrementing wait**. If true and `old_lr > min_lr + min_lr*1e-4`, set `lr=max(old_lr*factor,min_lr)` and reset wait=0. Then increment wait unconditionally in this non-improvement branch. Thus first reduction occurs on the sixth consecutive failed epoch; after reduction wait ends at 1, so next reduction may occur five epochs later. At the LR floor the wait counter continues growing. Cooldown=0 requires no special delay.
- EarlyStopping: patience=10, min_delta=0. Improvement strictly `current > best`; otherwise **check wait>=10 before increment** and set stop flag, then increment wait. Stop therefore occurs on the eleventh consecutive nonimproving epoch. No automatic best-weight restoration exists in this version; reload the separately saved best checkpoint for final evaluation.
- ModelCheckpoint: period=1, save_best_only=True, save_weights_only=True. Strictly greater accuracy saves; ties retain earlier epoch. It still runs after EarlyStopping sets its flag at that epoch.
- Keep independent best/wait values for plateau and early stopping. The 1e-4 vs 0 deltas can differ on a sufficiently large validation set. With 510 equally weighted samples, a one-sample accuracy increase exceeds 1e-4.

Relevant snapshot lines: callbacks.py ModelCheckpoint 378–418, EarlyStopping 449–508, ReduceLROnPlateau 838–916.

## Adam

Keras 2.0.8 `optimizers.py:400–451`: lr=.001, beta1=.9, beta2=.999, **epsilon=1e-8**, decay=0. Do not substitute backend epsilon (1e-7). No weight decay, clipping, AMSGrad, or learning-rate step decay is requested by Var-CNN.

For global update number t starting at 1:

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
lr_t = lr * sqrt(1-beta2**t)/(1-beta1**t)
p -= lr_t*m/(sqrt(v)+1e-8)
```

PyTorch standard Adam instead adds epsilon after bias correction of the variance. Equivalent denominator in the formula above is `sqrt(v)+eps_torch*sqrt(1-beta2**t)`. Consequently merely setting torch Adam eps=1e-8 does not implement Keras's exact formula. Prefer a minimal explicit optimizer matching the equation; alternatively change torch epsilon to `1e-8/sqrt(1-beta2**t)` at every global update with no weight decay, and document numerical limits. Float32 beta powers, reduction ordering and accelerator kernels can still yield cross-framework differences.

## Dataset iteration

Official preprocess_data.py shuffles per-site instances, unmonitored data, and the merged train/test sequence-label lists before writing HDF5. The resulting training order is fixed; data_generator.py does no shuffle. It slices contiguous batches, yields the final partial batch, then resets batch_start to zero after reaching the dataset end. `run_model.py` uses ceil(N/batch_size) steps and shuffle=False. There is no wrapping a partial batch with samples from the next pass and no drop_last.

For this project's fixed source150/valid510 adaptation, form one seed-controlled training permutation once at task initialization and reuse it for all epochs. This matches the once-permuted cycling behavior, not the exact original Python random preprocessing order; record this adaptation. Preserve fixed existing split and sample membership. Batch50 gives 306 full train batches for15300; validation gives10 full batches and one10-sample partial batch. Compute validation accuracy as total correct/510 (sample weighting), not unweighted average of11 batch accuracies. Validation order has no shuffle; original upstream preprocessing permutation is irrelevant for aggregate metrics, but saved predictions must retain manifest mapping.

## Exact epoch outline

1. Set model train mode, iterate fixed training order once, average training losses by sample count, apply KerasAdam on each batch; budget checks never manufacture epoch metrics for a partial epoch.
2. Set eval mode; evaluate all510 fixed validation samples, compute sample-weighted accuracy and macro-F1; accuracy alone controls callbacks.
3. Apply plateau update, then early-stop update, then strict-best checkpoint update, using independent callback state initialized at run start.
4. Log epoch metrics and LR before/after callbacks, save last checkpoint/callback state; if stop flagged end now. Otherwise repeat until150 epochs or explicit wall-time budget.
5. Reload best checkpoint for final train/valid predictions and independent metric verification. Save actual updates, examples presented, validation count, stop cause, elapsed time and final callback states.

A useful synthetic callback acceptance sequence is one improved epoch followed by11 ties: LR drops after tie6 and tie11, early stopping triggers on tie11, and only the first epoch is best. This catches modern callback off-by-one behavior.

## Implemented compatibility layer and checks

Run-local `keras_compat.py` implements KerasAdam and KerasCallbacks. `test_keras_compat.py` passed three CPU tests: first two optimizer updates against independent scalar calculations (including tiny gradients that expose epsilon placement), callback ties causing reduction on6/11 and stop on11, and separate improvement thresholds plus min-LR behavior. State serialization is checked. The optimizer uses Python double scalar beta powers with tensor moments in parameter dtype; Keras float32 scalar powers can have small numerical differences. The exact update equation is preserved, not bitwise parity. No training was performed in these checks.
