"""Post-run verification only; no training, selection changes or new data roles.

Added after completion and deliberately outside the original freeze manifest.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import RUN, ROOT, config, verify_freeze, deterministic, sha, save
from model import MultiViewModel, span_mask
from worker import predict, ema_update


def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(f1_score(y, p, labels=np.arange(102),
                                      average='macro', zero_division=0))}


def main():
    c = config(); verify_freeze(); deterministic('cuda', 2)
    assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    raw = torch.load(RUN/'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    assert set(raw) == set(c['roles'])
    historical = ROOT/'runs/20261003T030410Z_wide_bn_view_aligned_kd_43bb4cee'
    assert sha(RUN/'artifacts/prepared.pt') == sha(historical/'artifacts/prepared.pt')
    labels = {r: raw[r]['labels'].numpy() for r in c['roles']}
    data = {r: {k: v.cuda() for k, v in e.items() if k in ('timestamps','tam','labels')}
            for r, e in raw.items()}

    # Reproduce construction order without optimization: extra EMA construction
    # consumes CPU RNG but must not change CUDA RNG used by student dropout.
    torch.manual_seed(c['seeds'][0]); torch.cuda.manual_seed_all(c['seeds'][0])
    student = MultiViewModel('ce_only').cuda()
    cuda_rng = torch.cuda.get_rng_state().clone()
    ema = MultiViewModel('ce_only').cuda(); ema.load_state_dict(student.state_dict())
    ema.requires_grad_(False); ema.eval()
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    x = data['source']['timestamps'][:64]; clean = data['source']['tam'][:64]
    masked = span_mask(clean, torch.Generator().manual_seed(91731))
    with torch.no_grad():
        ema(x, masked); ema(x, clean)
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    # Verify the exact EMA update formula and buffer-copy rule without training.
    ep = next(ema.parameters()); p = next(student.parameters()); before = ep.clone()
    with torch.no_grad(): p.add_(0.01)
    expected = before * c['ema_consistency']['decay'] + p * (1-c['ema_consistency']['decay'])
    ema_update(ema, student, c['ema_consistency']['decay'])
    torch.testing.assert_close(ep, expected)
    assert all(torch.equal(a, b) for a, b in zip(ema.buffers(), student.buffers()))
    assert all(p.grad is None and not p.requires_grad for p in ema.parameters())
    del student, ema

    reports = []; metric_checks = logit_checks = replay = 0
    for seed in c['seeds']:
        initial_ref = torch.load(historical/'artifacts'/f'ce_only_{seed}'/'initial_state.pt', weights_only=True)
        index_ref = torch.load(historical/'artifacts'/f'ce_only_{seed}'/'index_stream.pt', weights_only=True)
        for kind in c['conditions']:
            key = f'{kind}_{seed}'; out = RUN/'artifacts'/key
            rep = json.loads((out/'report.json').read_text())
            assert rep['task'] == key and rep['seed'] == seed and rep['kind'] == kind
            assert rep['device'] == 'cuda' and rep['verified_roles'] == c['roles'] and not rep['future_access']
            assert rep['steps_completed'] == c['optimizer_steps']
            assert rep['validation_opportunities'] == len(c['eval_steps'])
            assert rep['ema_used'] == (kind != 'ce_only')
            assert rep['elapsed_seconds'] <= c['gpu_job_seconds']
            hist = json.loads((out/'history.json').read_text())
            assert [row['step'] for row in hist] == c['eval_steps']
            best = max(hist, key=lambda row: row['valid']['accuracy'])
            assert best['step'] == rep['best_step']
            assert all(np.isfinite(row[m]) for row in hist for m in ('train_loss','train_ce','train_consistency'))
            if kind != 'ce_only':
                assert rep['ema_decay'] == c['ema_consistency']['decay']
                assert all(row['train_consistency'] > 0 for row in hist)
            for phase, row in [('best', best), ('last', hist[-1])]:
                with np.load(out/f'predictions_{phase}.npz') as z:
                    for role in c['roles']:
                        pred = z[role]; assert pred.shape == labels[role].shape
                        assert np.isin(pred, np.arange(102)).all()
                        for m, val in metric(labels[role], pred).items():
                            assert abs(val-rep[phase][role][m]) < 1e-12
                            assert abs(val-row[role][m]) < 1e-12
                        metric_checks += 1
            initial = torch.load(out/'initial_state.pt', weights_only=True)
            assert set(initial) == set(initial_ref)
            assert all(torch.equal(initial[k], initial_ref[k]) for k in initial)
            assert torch.equal(torch.load(out/'index_stream.pt', weights_only=True), index_ref)
            grads = json.loads((out/'first_gradients.json').read_text())
            active = [v for k,v in grads.items() if (k.startswith('generator.stages') or k == 'generator.to_features.weight') and k.endswith('weight')]
            assert active and all(v is not None and np.isfinite(v) and v > 0 for v in active)
            ck = torch.load(RUN/'checkpoints'/f'{key}_best.pt', weights_only=True, map_location='cuda')
            assert ck['step'] == rep['best_step'] and ck['config_sha256'] == sha(RUN/'config.json')
            model = MultiViewModel('ce_only').cuda(); model.load_state_dict(ck['state_dict'])
            assert sum(p.numel() for p in model.parameters()) == rep['parameters_total']
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == rep['parameters_trainable']
            assert all(int(v) == rep['best_step'] for k,v in model.named_buffers() if k.endswith('num_batches_tracked'))
            with np.load(out/'logits_best.npz') as z, np.load(out/'predictions_best.npz') as preds:
                for role in c['roles']:
                    saved = z[role]
                    assert saved.shape == (len(labels[role]),102) and np.isfinite(saved).all()
                    assert np.array_equal(saved.argmax(1),preds[role]); logit_checks += 1
                    fresh = predict(model,data[role],c['eval_batch_size'])
                    assert np.array_equal(fresh.argmax(1),preds[role])
                    np.testing.assert_allclose(fresh,saved,rtol=1e-5,atol=1e-5); replay += 1
                    if kind == 'ce_only':
                        with np.load(historical/'artifacts'/key/'logits_best.npz') as old:
                            assert np.array_equal(saved.argmax(1),old[role].argmax(1))
            del model; reports.append(rep)
            print(json.dumps({'audited':key,'checkpoint_replay_groups':replay}),flush=True)

    means = {k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k]))
                for m in ('accuracy','macro_f1')} for k in c['conditions']}
    existing = json.loads((RUN/'artifacts/summary.json').read_text())
    for k in means:
        for m,v in means[k].items(): assert abs(v-existing['means'][k][m]) < 1e-12
    comparisons = {}
    def score(k,s,m): return next(r['best']['valid'][m] for r in reports if r['kind']==k and r['seed']==s)
    for k in c['conditions'][1:]:
        da = [score(k,s,'accuracy')-score('ce_only',s,'accuracy') for s in c['seeds']]
        df = [score(k,s,'macro_f1')-score('ce_only',s,'macro_f1') for s in c['seeds']]
        comparisons[k+' - ce_only'] = {'accuracy_mean_delta_pp':float(np.mean(da)*100),
            'accuracy_per_seed_pp':[float(v*100) for v in da], 'macro_f1_mean_delta_pp':float(np.mean(df)*100),
            'positive_seed_count':sum(v>0 for v in da), 'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
    assert comparisons == existing['comparisons']
    audit = {'passed':True,'training_tasks_completed':len(reports),'freeze_hashes_verified':len(json.loads((RUN/'artifacts/freeze.json').read_text())),
             'metric_groups_verified':metric_checks,'logit_groups_verified':logit_checks,
             'checkpoint_replay_groups_verified':replay,'initialization_groups_verified':len(reports),
             'index_streams_verified':len(reports),'history_selection_groups_verified':len(reports),
             'cuda_rng_preserved_by_ema_construction_and_eval':True,'ema_update_formula_verified':True,
             'historical_ce_predictions_reproduced':True,'means':means,'comparisons':comparisons,
             'confidence_gate_used':False,'inference_model':'online student, not EMA',
             'full_ema_trajectory_saved':False,'future_access':False,
             'audit_script_sha256':sha(RUN/'postrun_audit.py')}
    save(RUN/'artifacts/postrun_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__': main()
