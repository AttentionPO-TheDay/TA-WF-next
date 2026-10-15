"""Completion audit only; no optimization or new selection opportunities.

Added after training, outside the original freeze manifest.  Verify historical
copies separately from the six new tasks, and replay source/valid checkpoints.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import RUN, ROOT, config, verify_freeze, deterministic, save, sha
from model import MultiViewModel, span_mask
from worker import predict, js_divergence


def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(f1_score(y, p, labels=np.arange(102),
                                      average='macro', zero_division=0))}


def main():
    c = config(); verify_freeze(); deterministic('cuda', 2)
    assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    historical = ROOT/c['historical_reference_run']
    assert sha(RUN/'artifacts/prepared.pt') == sha(historical/'artifacts/prepared.pt')
    raw = torch.load(RUN/'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    assert set(raw) == set(c['roles'])
    labels = {r: raw[r]['labels'].numpy() for r in c['roles']}
    data = {r: {k: v.cuda() for k, v in e.items() if k in ('timestamps', 'tam', 'labels')}
            for r, e in raw.items()}
    fixed = span_mask(data['valid']['tam'], torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']))
    preflight = json.loads((RUN/'artifacts/preflight.json').read_text())
    assert preflight['passed'] and preflight['auxiliary_logits_direct_ce_gradient_zero']
    assert preflight['peer_target_detached_derivative_verified'] and preflight['one_bn_update_verified']
    all_reports = []; counters = {'metric_groups': 0, 'logit_groups': 0, 'replay_groups': 0,
                                 'diagnostic_groups': 0, 'initializations': 0, 'index_streams': 0,
                                 'histories': 0, 'reference_files': 0}
    for seed in c['seeds']:
        initial_ref = torch.load(RUN/'artifacts'/f'ce_only_{seed}'/'initial_state.pt', weights_only=True)
        index_ref = torch.load(RUN/'artifacts'/f'ce_only_{seed}'/'index_stream.pt', weights_only=True)
        for kind in c['reference_conditions'] + c['conditions']:
            key = f'{kind}_{seed}'; out = RUN/'artifacts'/key
            is_new = kind in c['conditions']
            rep = json.loads((out/'report.json').read_text())
            assert rep['task'] == key and rep['kind'] == kind and rep['seed'] == seed
            assert rep['device'] == 'cuda' and rep['verified_roles'] == c['roles'] and not rep['future_access']
            assert rep['steps_completed'] == c['optimizer_steps']
            assert rep['validation_opportunities'] == len(c['eval_steps'])
            if is_new:
                assert rep['primary_view_ce_only'] and rep['single_network']
                assert rep['peer_consistency'] == (kind == 'primary_ce_peer')
                assert rep['elapsed_seconds'] <= c['gpu_job_seconds']
            else:
                for p in out.iterdir():
                    if p.is_file():
                        assert sha(p) == sha(historical/'artifacts'/key/p.name)
                        counters['reference_files'] += 1
                assert sha(RUN/'checkpoints'/f'{key}_best.pt') == sha(historical/'checkpoints'/f'{key}_best.pt')
                counters['reference_files'] += 1
            history = json.loads((out/'history.json').read_text())
            assert [r['step'] for r in history] == c['eval_steps']
            best = max(history, key=lambda r: r['valid']['accuracy'])
            assert best['step'] == rep['best_step']
            counters['histories'] += 1
            if is_new:
                previous = 0
                for row in history:
                    assert all(np.isfinite(row[m]) for m in ('train_loss', 'train_ce', 'train_peer'))
                    ramp = c['peer_consistency']['ramp_steps']
                    weights = [0. if kind == 'primary_ce_only_joint' else
                               .25 * min(max(t-ramp, 0)/ramp, 1.)
                               for t in range(previous+1, row['step']+1)]
                    assert abs(row['mean_peer_weight']-np.mean(weights)) < 1e-12
                    if kind == 'primary_ce_only_joint': assert row['train_peer'] == 0.
                    previous = row['step']
                grads = json.loads((out/'first_gradients.json').read_text())
                active = [v for k,v in grads.items() if
                          (k.startswith('generator.stages') or k == 'generator.to_features.weight') and k.endswith('weight')]
                assert active and all(v is not None and np.isfinite(v) and v > 0 for v in active)
            initial = torch.load(out/'initial_state.pt', weights_only=True)
            assert set(initial) == set(initial_ref) and all(torch.equal(v, initial_ref[k]) for k,v in initial.items())
            assert torch.equal(torch.load(out/'index_stream.pt', weights_only=True), index_ref)
            counters['initializations'] += 1; counters['index_streams'] += 1
            for phase, row in [('best', best), ('last', history[-1])]:
                with np.load(out/f'predictions_{phase}.npz') as z:
                    for role in c['roles']:
                        p = z[role]; assert p.shape == labels[role].shape
                        assert np.isin(p, np.arange(102)).all()
                        for m, val in metric(labels[role], p).items():
                            assert abs(val-rep[phase][role][m]) < 1e-12
                            assert abs(val-row[role][m]) < 1e-12
                        counters['metric_groups'] += 1
            ck = torch.load(RUN/'checkpoints'/f'{key}_best.pt', weights_only=True, map_location='cuda')
            assert ck['step'] == rep['best_step']
            if is_new: assert ck['config_sha256'] == sha(RUN/'config.json')
            model = MultiViewModel('ce_only').cuda(); model.load_state_dict(ck['state_dict'])
            assert sum(p.numel() for p in model.parameters()) == rep['parameters_total']
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == rep['parameters_trainable']
            assert all(int(v) == rep['best_step'] for k,v in model.named_buffers() if k.endswith('num_batches_tracked'))
            with np.load(out/'logits_best.npz') as z, np.load(out/'predictions_best.npz') as p:
                for role in c['roles']:
                    saved = z[role]; assert saved.shape == (len(labels[role]), 102) and np.isfinite(saved).all()
                    assert np.array_equal(saved.argmax(1), p[role])
                    fresh = predict(model, data[role], c['eval_batch_size'])
                    assert np.array_equal(fresh.argmax(1), p[role])
                    np.testing.assert_allclose(fresh, saved, rtol=1e-5, atol=1e-5)
                    counters['logit_groups'] += 1; counters['replay_groups'] += 1
                if is_new:
                    cl = torch.from_numpy(z['valid'].copy())
                    ml = torch.from_numpy(predict(model, {**data['valid'], 'tam': fixed}, c['eval_batch_size']))
                    diag = {'clean_accuracy': metric(labels['valid'], cl.argmax(1).numpy())['accuracy'],
                            'masked_accuracy': metric(labels['valid'], ml.argmax(1).numpy())['accuracy'],
                            'prediction_agreement': float((cl.argmax(1) == ml.argmax(1)).float().mean()),
                            'js_divergence': js_divergence(cl, ml)}
                    saved_diag = json.loads((out/'fixed_view_diagnostic.json').read_text())
                    assert saved_diag == rep['fixed_view_diagnostic']
                    assert all(abs(diag[k]-saved_diag[k]) < 1e-7 for k in diag)
                    counters['diagnostic_groups'] += 1
            del model; all_reports.append(rep)
            print(json.dumps({'audited': key, 'replay_groups': counters['replay_groups']}), flush=True)
    summary = json.loads((RUN/'artifacts/summary.json').read_text())
    def score(k, s, m):
        return next(r['best']['valid'][m] for r in all_reports if r['kind'] == k and r['seed'] == s)
    means = {k: {m: float(np.mean([score(k,s,m) for s in c['seeds']]))
                 for m in ('accuracy','macro_f1')} for k in c['reference_conditions']+c['conditions']}
    for k in means:
        for m,v in means[k].items(): assert abs(v-summary['means'][k][m]) < 1e-12
    comparisons = {}
    for key in summary['comparisons']:
        a,b = key.split(' - ')
        da = [score(a,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']]
        df = [score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]
        comparisons[key] = {'accuracy_mean_delta_pp':float(np.mean(da)*100),
                            'accuracy_per_seed_pp':[float(v*100) for v in da],
                            'macro_f1_mean_delta_pp':float(np.mean(df)*100),
                            'positive_seed_count':sum(v>0 for v in da),
                            'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
    assert comparisons == summary['comparisons']
    diag_means = {k:{m:float(np.mean([r['fixed_view_diagnostic'][m] for r in all_reports if r['kind']==k]))
                     for m in ('masked_accuracy','prediction_agreement','js_divergence')} for k in c['conditions']}
    audit = {'passed':True, 'training_tasks_completed':6, 'historical_reference_tasks':9,
             'freeze_hashes_verified':len(json.loads((RUN/'artifacts/freeze.json').read_text())),
             'checks':counters, 'means':means, 'comparisons':comparisons,
             'new_condition_diagnostic_means':diag_means,
             'new_preflight_loss_gradients_verified':True,
             'original_training_code_modified':False, 'future_access':False,
             'audit_script_sha256':sha(RUN/'postrun_audit.py')}
    save(RUN/'artifacts/postrun_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
