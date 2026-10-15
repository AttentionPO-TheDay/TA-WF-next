"""Read-only completion audit added after training, outside original freeze."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import RUN, ROOT, config, verify_freeze, deterministic, save, sha
from model import MultiViewModel, span_mask
from worker import predict


def metric(y, p):
    return {'accuracy':float(accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0))}


def main():
    c=config(); verify_freeze()
    assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    deterministic('cuda',2)
    historical=ROOT/c['historical_reference_run']
    assert sha(RUN/'artifacts/prepared.pt')==sha(historical/'artifacts/prepared.pt')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
    assert set(raw)==set(c['roles'])
    labels={r:raw[r]['labels'].numpy() for r in c['roles']}
    data={r:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}
    fixed=span_mask(data['valid']['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']))
    reports=[]; checks={'metric_groups':0,'logit_groups':0,'checkpoint_replay_groups':0,
                        'fixed_view_diagnostic_groups':0,'initialization_groups':0,
                        'index_streams':0,'history_selection_groups':0,'historical_files':0,
                        'new_perturbation_histories':0,'new_bn_and_restore_reports':0}
    for seed in c['seeds']:
        ref=RUN/'artifacts'/f'ce_only_{seed}'
        initref=torch.load(ref/'initial_state.pt',weights_only=True)
        indexref=torch.load(ref/'index_stream.pt',weights_only=True)
        for kind in ['ce_only']+c['conditions']:
            key=f'{kind}_{seed}'; out=RUN/'artifacts'/key; is_new=kind!='ce_only'
            rep=json.loads((out/'report.json').read_text())
            assert rep['task']==key and rep['kind']==kind and rep['seed']==seed
            assert rep['device']=='cuda' and rep['verified_roles']==c['roles'] and not rep['future_access']
            assert rep['steps_completed']==c['optimizer_steps']
            assert rep['validation_opportunities']==len(c['eval_steps'])
            hist=json.loads((out/'history.json').read_text())
            assert [row['step'] for row in hist]==c['eval_steps']
            best=max(hist,key=lambda row:row['valid']['accuracy'])
            assert best['step']==rep['best_step']; checks['history_selection_groups']+=1
            if is_new:
                assert rep['sam_used']==(kind=='sam_ce') and rep['rho']==c['sam']['rho']
                assert rep['bn_buffer_restore_checks']==c['optimizer_steps']
                assert rep['parameter_restore_checks']==(c['optimizer_steps'] if kind=='sam_ce' else 0)
                assert rep['elapsed_seconds']<=c['gpu_job_seconds']
                for row in hist:
                    assert all(np.isfinite(row[m]) for m in ('first_loss','second_loss','grad_norm','perturb_norm'))
                    assert row['grad_norm']>0 and row['train_loss']==row['second_loss']
                    assert abs(row['perturb_norm']-(c['sam']['rho'] if kind=='sam_ce' else 0))<1e-6
                checks['new_perturbation_histories']+=1; checks['new_bn_and_restore_reports']+=1
            else:
                for p in out.iterdir():
                    if p.is_file():
                        assert sha(p)==sha(historical/'artifacts'/key/p.name); checks['historical_files']+=1
                assert sha(RUN/'checkpoints'/f'{key}_best.pt')==sha(historical/'checkpoints'/f'{key}_best.pt')
                checks['historical_files']+=1
            ini=torch.load(out/'initial_state.pt',weights_only=True)
            assert set(ini)==set(initref) and all(torch.equal(v,initref[k]) for k,v in ini.items())
            assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True),indexref)
            checks['initialization_groups']+=1; checks['index_streams']+=1
            for phase,row in [('best',best),('last',hist[-1])]:
                with np.load(out/f'predictions_{phase}.npz') as z:
                    for role in c['roles']:
                        p=z[role]; assert p.shape==labels[role].shape and np.isin(p,np.arange(102)).all()
                        for m,v in metric(labels[role],p).items():
                            assert abs(v-rep[phase][role][m])<1e-12 and abs(v-row[role][m])<1e-12
                        checks['metric_groups']+=1
            ck=torch.load(RUN/'checkpoints'/f'{key}_best.pt',weights_only=True,map_location='cuda')
            assert ck['step']==rep['best_step']
            if is_new: assert ck['config_sha256']==sha(RUN/'config.json')
            model=MultiViewModel('ce_only').cuda(); model.load_state_dict(ck['state_dict'])
            assert sum(p.numel() for p in model.parameters())==rep['parameters_total']
            assert sum(p.numel() for p in model.parameters() if p.requires_grad)==rep['parameters_trainable']
            assert all(int(v)==rep['best_step'] for k,v in model.named_buffers() if k.endswith('num_batches_tracked'))
            with np.load(out/'logits_best.npz') as z, np.load(out/'predictions_best.npz') as pr:
                for role in c['roles']:
                    saved=z[role]; assert saved.shape==(len(labels[role]),102) and np.isfinite(saved).all()
                    assert np.array_equal(saved.argmax(1),pr[role])
                    fresh=predict(model,data[role],c['eval_batch_size'])
                    assert np.array_equal(fresh.argmax(1),pr[role])
                    np.testing.assert_allclose(fresh,saved,rtol=1e-5,atol=1e-5)
                    checks['logit_groups']+=1; checks['checkpoint_replay_groups']+=1
                if is_new:
                    clean=z['valid']; masked=predict(model,{**data['valid'],'tam':fixed},c['eval_batch_size'])
                    diag={'clean_accuracy':metric(labels['valid'],clean.argmax(1))['accuracy'],
                          'masked_accuracy':metric(labels['valid'],masked.argmax(1))['accuracy'],
                          'prediction_agreement':float((clean.argmax(1)==masked.argmax(1)).mean())}
                    assert diag==rep['fixed_view_diagnostic']==json.loads((out/'fixed_view_diagnostic.json').read_text())
                    checks['fixed_view_diagnostic_groups']+=1
            reports.append(rep); del model
            print(json.dumps({'audited':key,'checkpoint_replay_groups':checks['checkpoint_replay_groups']}),flush=True)
    def score(k,s,m): return next(r['best']['valid'][m] for r in reports if r['kind']==k and r['seed']==s)
    summary=json.loads((RUN/'artifacts/summary.json').read_text())
    means={k:{m:float(np.mean([score(k,s,m) for s in c['seeds']])) for m in ('accuracy','macro_f1')}
           for k in ['ce_only']+c['conditions']}
    for k in means:
        for m,v in means[k].items(): assert abs(v-summary['means'][k][m])<1e-12
    comparisons={}
    for key in summary['comparisons']:
        a,b=key.split(' - ')
        da=[score(a,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']]
        df=[score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]
        comparisons[key]={'accuracy_mean_delta_pp':float(np.mean(da)*100),
                          'accuracy_per_seed_pp':[float(v*100) for v in da],
                          'macro_f1_mean_delta_pp':float(np.mean(df)*100),
                          'positive_seed_count':sum(v>0 for v in da),
                          'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
    assert comparisons==summary['comparisons']
    audit={'passed':True,'freeze_hashes_verified':len(json.loads((RUN/'artifacts/freeze.json').read_text())),
           'training_tasks_completed':6,'historical_reference_tasks':3,'checks':checks,
           'means':means,'comparisons':comparisons,'bn_rule':c['sam']['bn_rule'],
           'rho':c['sam']['rho'],'full_training_replayed':False,
           'future_access':False,'audit_script_sha256':sha(RUN/'postrun_audit.py')}
    save(RUN/'artifacts/postrun_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
