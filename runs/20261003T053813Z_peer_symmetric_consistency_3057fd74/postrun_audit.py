"""Read-only audit for single-network two-view peer consistency."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import RUN, ROOT, config, verify_freeze, deterministic, save, sha
from model import MultiViewModel, span_mask, symmetric_peer_loss
from worker import predict, js_divergence

def metric(y,p):
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0))}

def main():
    c=config(); verify_freeze(); deterministic('cuda',2); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    historical=ROOT/'runs/20261003T043438Z_ema_view_consistency_ef5e1766'; raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); oldraw=torch.load(historical/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); assert sha(RUN/'artifacts/prepared.pt')==sha(historical/'artifacts/prepared.pt')
    labels={r:raw[r]['labels'].numpy() for r in c['roles']}; data={r:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}
    kinds=['dual_mask_ce','mask_peer','dual_cross_ce','cross_peer']; reports=[]; metric_checks=logit_checks=replay=diag_checks=init_checks=index_checks=history_checks=0
    for seed in c['seeds']:
        refi=torch.load(RUN/'artifacts'/f'ce_only_{seed}'/'initial_state.pt',weights_only=True); refidx=torch.load(RUN/'artifacts'/f'ce_only_{seed}'/'index_stream.pt',weights_only=True)
        for kind in kinds:
            key=f'{kind}_{seed}'; out=RUN/'artifacts'/key; rep=json.loads((out/'report.json').read_text()); assert rep['task']==key and rep['single_network'] and rep['verified_roles']==c['roles'] and not rep['future_access']; assert rep['steps_completed']==c['optimizer_steps'] and rep['validation_opportunities']==len(c['eval_steps'])
            hist=json.loads((out/'history.json').read_text()); assert [x['step'] for x in hist]==c['eval_steps']; best=max(hist,key=lambda x:x['valid']['accuracy']); assert best['step']==rep['best_step']; history_checks+=1
            ini=torch.load(out/'initial_state.pt',weights_only=True); idx=torch.load(out/'index_stream.pt',weights_only=True); assert all(torch.equal(ini[k],refi[k]) for k in ini); assert torch.equal(idx,refidx); init_checks+=1; index_checks+=1
            assert all(np.isfinite(x[m]) for x in hist for m in ('train_loss','train_ce','train_peer'))
            if kind in ('dual_mask_ce','dual_cross_ce'):
                assert all(x['mean_peer_weight'] == 0 for x in hist)
            else:
                assert_peer_weights(hist, c)
            for phase,row in [('best',best),('last',hist[-1])]:
                with np.load(out/f'predictions_{phase}.npz') as z:
                    for role in c['roles']:
                        p=z[role]; got=metric(labels[role],p); assert p.shape==labels[role].shape and np.isin(p,np.arange(102)).all(); assert all(abs(got[m]-rep[phase][role][m])<1e-12 and abs(got[m]-row[role][m])<1e-12 for m in got); metric_checks+=1
            ck=torch.load(RUN/'checkpoints'/f'{key}_best.pt',weights_only=True,map_location='cuda'); assert ck['step']==rep['best_step'] and ck['config_sha256']==sha(RUN/'config.json'); model=MultiViewModel('ce_only').cuda(); model.load_state_dict(ck['state_dict']); assert sum(p.numel() for p in model.parameters())==rep['parameters_total']
            with np.load(out/'logits_best.npz') as z, np.load(out/'predictions_best.npz') as pr:
                for role in c['roles']:
                    saved=z[role]; assert saved.shape==(len(labels[role]),102) and np.isfinite(saved).all() and np.array_equal(saved.argmax(1),pr[role]); fresh=predict(model,data[role],c['eval_batch_size']); assert np.array_equal(fresh.argmax(1),pr[role]); np.testing.assert_allclose(fresh,saved,rtol=1e-5,atol=1e-5); logit_checks+=1; replay+=1
            d=json.loads((out/'fixed_view_diagnostic.json').read_text()); ve=data['valid']; fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed'])); model.eval()
            with torch.inference_mode(): cl=model(ve['timestamps'],ve['tam']); ml=model(ve['timestamps'],fixed)
            got={'clean_accuracy':metric(labels['valid'],cl.argmax(1).cpu().numpy())['accuracy'],'masked_accuracy':metric(labels['valid'],ml.argmax(1).cpu().numpy())['accuracy'],'prediction_agreement':float((cl.argmax(1)==ml.argmax(1)).float().mean()),'js_divergence':js_divergence(cl,ml)}; assert all(abs(got[k]-d[k])<1e-6 for k in got); diag_checks+=1; del model; reports.append(rep)
    # Historical CE reference predictions are unchanged.
    for seed in c['seeds']:
        with np.load(RUN/'artifacts'/f'ce_only_{seed}'/'predictions_best.npz') as a, np.load(historical/'artifacts'/f'ce_only_{seed}'/'predictions_best.npz') as b:
            for r in c['roles']: assert np.array_equal(a[r],b[r])
    allrep=[json.loads((RUN/'artifacts'/f'ce_only_{s}'/'report.json').read_text())|{'kind':'ce_only','seed':s} for s in c['seeds']]+reports
    def score(k,s,m): return next(r['best']['valid'][m] for r in allrep if r['kind']==k and r['seed']==s)
    comps=json.loads((RUN/'artifacts/summary.json').read_text())['comparisons']; calc={}
    for k,b in [('mask_peer','dual_mask_ce'),('cross_peer','dual_cross_ce'),('dual_mask_ce','ce_only'),('dual_cross_ce','ce_only'),('mask_peer','ce_only'),('cross_peer','ce_only')]:
        da=[score(k,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']]; df=[score(k,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]; calc[f'{k} - {b}']={'accuracy_mean_delta_pp':float(np.mean(da)*100),'accuracy_per_seed_pp':[float(x*100) for x in da],'macro_f1_mean_delta_pp':float(np.mean(df)*100),'positive_seed_count':int(sum(x>0 for x in da)),'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
    assert calc==comps; audit={'passed':True,'freeze_hashes_verified':len(json.loads((RUN/'artifacts/freeze.json').read_text())),'training_tasks_completed':len(reports),'metric_groups_verified':metric_checks,'logit_groups_verified':logit_checks,'checkpoint_replay_groups_verified':replay,'fixed_view_diagnostic_groups_verified':diag_checks,'history_selection_groups_verified':history_checks,'initialization_groups_verified':init_checks,'index_streams_verified':index_checks,'historical_ce_predictions_reproduced':True,'single_network_verified':True,'fixed_teacher_loaded':False,'ema_loaded':False,'comparisons':calc,'future_access':False}; save(RUN/'artifacts/postrun_audit.json',audit); print(json.dumps(audit,ensure_ascii=False,indent=2))

def assert_peer_weights(hist,c):
    assert all(0 <= x['mean_peer_weight'] <= c['peer_consistency']['max_weight'] for x in hist)

if __name__=='__main__': main()
