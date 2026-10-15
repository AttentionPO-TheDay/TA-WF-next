"""Post-run verification only; no optimization or new data roles."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import RUN, ROOT, config, verify_freeze, deterministic, save, sha
from model import MultiViewModel
from worker import predict, teacher_source


def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, labels=np.arange(102), average='macro', zero_division=0))}


def main():
    c = config(); verify_freeze()
    assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    deterministic('cuda', 2)
    raw = torch.load(RUN/'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    historical = ROOT/c['historical_teacher_run']
    assert sha(RUN/'artifacts/prepared.pt') == sha(historical/'artifacts/prepared.pt')
    labels = {r: raw[r]['labels'].numpy() for r in c['roles']}
    data = {r: {k: v.cuda() for k, v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}
    for seed in c['seeds']:
        assert sha(RUN/'artifacts'/f'teacher_logits_{seed}.npz') == sha(historical/'artifacts'/f'wide_bn_{seed}'/'logits_best.npz')
        ref = torch.load(historical/'artifacts'/f'wide_bn_{seed}'/'initial_state.pt', weights_only=True, map_location='cpu')
        local = torch.load(RUN/'artifacts'/f'teacher_initial_{seed}.pt', weights_only=True, map_location='cpu')
        assert set(ref)==set(local) and all(torch.equal(v, local[k]) for k,v in ref.items())
        original = teacher_source(seed, 'ensemble_kd')
        perm = torch.randperm(len(original), generator=torch.Generator().manual_seed(seed+70000))
        assert torch.equal(teacher_source(seed,'shuffled_kd'), original[perm])
    reports = []; checks = replay = logits_checks = 0
    for seed in c['seeds']:
        initial_ref = torch.load(historical/'artifacts'/f'wide_bn_{seed}'/'initial_state.pt', weights_only=True, map_location='cpu')
        index_ref = torch.load(historical/'artifacts'/f'wide_bn_{seed}'/'index_stream.pt', weights_only=True, map_location='cpu')
        for kind in c['conditions']:
            key=f'{kind}_{seed}'; out=RUN/'artifacts'/key
            rep=json.loads((out/'report.json').read_text())
            assert rep['kind']==kind and rep['seed']==seed and rep['device']=='cuda'
            assert rep['steps_completed']==c['optimizer_steps'] and rep['validation_opportunities']==len(c['eval_steps'])
            assert rep['verified_roles']==c['roles'] and not rep['future_access'] and rep['bn_eval_buffers_unchanged']
            hist=json.loads((out/'history.json').read_text())
            assert [r['step'] for r in hist]==c['eval_steps']
            best=max(hist,key=lambda r:r['valid']['accuracy'])
            assert best['step']==rep['best_step']
            for phase,row in [('best',best),('last',hist[-1])]:
                for role in c['roles']:
                    for m in ['accuracy','macro_f1']: assert abs(row[role][m]-rep[phase][role][m])<1e-12
                with np.load(out/f'predictions_{phase}.npz') as z:
                    for role in c['roles']:
                        pred=z[role]; assert pred.shape==labels[role].shape and np.isin(pred,np.arange(102)).all()
                        got=metric(labels[role],pred)
                        for m in got: assert abs(got[m]-rep[phase][role][m])<1e-12
                        checks+=1
            state=torch.load(out/'initial_state.pt',weights_only=True,map_location='cpu')
            assert set(state)==set(initial_ref) and all(torch.equal(v,state[k]) for k,v in initial_ref.items())
            assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True,map_location='cpu'),index_ref)
            ck=torch.load(RUN/'checkpoints'/f'{key}_best.pt',weights_only=True,map_location='cpu')
            assert ck['step']==rep['best_step'] and ck['config_sha256']==sha(RUN/'config.json')
            model=MultiViewModel(kind).cuda(); model.load_state_dict(ck['state_dict'])
            assert sum(p.numel() for p in model.parameters())==rep['parameters_total']
            assert sum(p.numel() for p in model.parameters() if p.requires_grad)==rep['parameters_trainable']
            assert all(int(v)==rep['best_step'] for k,v in model.named_buffers() if k.endswith('num_batches_tracked'))
            assert all(torch.equal(p.cpu(),state[k]) for k,p in model.named_parameters() if not p.requires_grad)
            with np.load(out/'logits_best.npz') as z, np.load(out/'predictions_best.npz') as preds:
                for role in c['roles']:
                    saved=z[role]; assert saved.shape==(len(labels[role]),102) and np.isfinite(saved).all()
                    assert np.array_equal(saved.argmax(1),preds[role]); logits_checks+=1
                    fresh=predict(model,data[role],c['eval_batch_size'])
                    assert np.array_equal(fresh.argmax(1),preds[role])
                    np.testing.assert_allclose(fresh,saved,rtol=1e-5,atol=1e-5); replay+=1
            if kind=='ce_only':
                with np.load(historical/'artifacts'/f'wide_bn_{seed}'/'logits_best.npz') as old, np.load(out/'logits_best.npz') as new:
                    for role in c['roles']:
                        assert np.array_equal(old[role].argmax(1),new[role].argmax(1))
            del model; reports.append(rep)
            print(json.dumps({'audited':key,'replay_groups':replay}),flush=True)
    def score(k,s,m): return next(r['best']['valid'][m] for r in reports if r['kind']==k and r['seed']==s)
    means={k:{m:float(np.mean([score(k,s,m) for s in c['seeds']])) for m in ('accuracy','macro_f1')} for k in c['conditions']}
    comparisons={}
    for a,b in [('ensemble_kd','ce_only'),('shuffled_kd','ce_only'),('ensemble_kd','shuffled_kd')]:
        da=[score(a,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']]
        df=[score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]
        comparisons[f'{a} - {b}']={'accuracy_mean_delta_pp':float(np.mean(da)*100),'accuracy_per_seed_pp':[float(v*100) for v in da],'macro_f1_mean_delta_pp':float(np.mean(df)*100),'positive_seed_count':sum(v>0 for v in da),'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
    old=json.loads((RUN/'artifacts/summary.json').read_text())
    for k,v in means.items():
        for m in v: assert abs(v[m]-old['means'][k][m])<1e-12
    summary={'passed':True,'means':means,'comparisons':comparisons,'metric_groups_verified':checks,'logit_groups_verified':logits_checks,'checkpoint_replay_groups_verified':replay,'freeze_hashes_verified':len(json.loads((RUN/'artifacts/freeze.json').read_text())),'initialization_groups_verified':9,'index_streams_verified':9,'history_selection_groups_verified':9,'historical_ce_predictions_reproduced':True,'teacher_provenance_verified':True,'shuffled_permutation_verified':True,'training_tasks_completed':9,'future_access':False,'reports':reports}
    save(RUN/'artifacts/postrun_audit.json',summary)
    print(json.dumps({'means':means,'comparisons':comparisons,'checks':checks,'replay':replay},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
