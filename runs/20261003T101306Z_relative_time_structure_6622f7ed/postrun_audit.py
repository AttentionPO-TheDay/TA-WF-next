import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import torch
from common import RUN,ROOT,config,verify_freeze,deterministic,save,sha,metric
from model import MultiViewModel,span_mask
from worker import predict

def main():
    c=config();verify_freeze();deterministic('cuda',2)
    ancestor='49b01254108433528398d17fd5a4df9618a7632ba03c0a63864236786e9fc4db';assert sha(RUN/'artifacts/prepared.pt')==ancestor
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert set(raw)=={'source','valid'}
    data={r:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}; summary=json.loads((RUN/'artifacts/summary.json').read_text()); reports=[]; checks={'metrics':0,'logits_and_checkpoints':0,'fixed_mask':0,'initialization':0,'index_streams':0,'selection':0,'feature_reference':0}
    models={k:MultiViewModel(k).cuda() for k in c['conditions']}
    for seed in c['seeds']:
        for kind in c['conditions']:
            out=RUN/'artifacts'/f'{kind}_{seed}';report=json.loads((out/'report.json').read_text());reports.append(report);assert report['steps_completed']==12800 and not report['future_access']
            history=json.loads((out/'history.json').read_text());best=max(history,key=lambda h:h['valid']['accuracy']);assert best['step']==report['best_step'];checks['selection']+=1
            old=ROOT/c['historical_reference_run']/'artifacts'/f'ce_only_{seed}'
            ini=torch.load(out/'initial_state.pt',weights_only=True);ref=torch.load(old/'initial_state.pt',weights_only=True)
            assert set(ref).issubset(ini) and all(torch.equal(ini[k],ref[k]) for k in ref);checks['initialization']+=1
            peer=torch.load(RUN/'artifacts'/f'relative_time_{seed}'/'initial_state.pt',weights_only=True)
            assert ini.keys()==peer.keys() and all(torch.equal(ini[k],peer[k]) for k in ini)
            assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True),torch.load(old/'index_stream.pt',weights_only=True));checks['index_streams']+=1
            ck=torch.load(RUN/'checkpoints'/f'{kind}_{seed}_best.pt',weights_only=True,map_location='cuda');assert ck['config_sha256']==sha(RUN/'config.json');models[kind].load_state_dict(ck['state_dict']);logits=np.load(out/'logits_best.npz')
            for role,e in data.items():
                z=logits[role];assert z.shape==(len(e['labels']),102) and np.isfinite(z).all();score=metric(e['labels'].cpu().numpy(),z.argmax(1));assert all(np.isclose(score[m],report['best'][role][m],atol=1e-12) for m in score);checks['metrics']+=1
                replay=predict(models[kind],e,c['eval_batch_size']);assert np.allclose(replay,z,atol=1e-5,rtol=1e-5) and np.array_equal(replay.argmax(1),z.argmax(1));checks['logits_and_checkpoints']+=1
            ve=data['valid'];fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']));mz=predict(models[kind],{**ve,'tam':fixed},c['eval_batch_size']);score=metric(ve['labels'].cpu().numpy(),mz.argmax(1));assert np.isclose(score['accuracy'],report['fixed_view_diagnostic']['masked_accuracy']);checks['fixed_mask']+=1
            feat=MultiViewModel.relative_time_features(ve['timestamps']);assert np.isclose(feat.mean().item(),report['fixed_view_diagnostic']['relative_feature_mean']);checks['feature_reference']+=1
        print(json.dumps({'audited_seed':seed}),flush=True)
    means={k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k])) for m in ('accuracy','macro_f1')} for k in c['conditions']};assert means==summary['means']
    a=[r for r in reports if r['kind']=='relative_time'];b=[r for r in reports if r['kind']=='ce_control'];da=[100*(x['best']['valid']['accuracy']-y['best']['valid']['accuracy']) for x,y in zip(a,b)];df=[100*(x['best']['valid']['macro_f1']-y['best']['valid']['macro_f1']) for x,y in zip(a,b)];comp=summary['comparison_relative_time_minus_ce_control'];assert np.allclose(da,comp['accuracy_per_seed_pp']);assert np.isclose(np.mean(da),comp['accuracy_mean_delta_pp']);assert np.isclose(np.mean(df),comp['macro_f1_mean_delta_pp']);assert comp['positive_seed_count']==sum(v>0 for v in da);assert comp['pass']==bool(np.mean(da)>=1 and min(da)>0 and np.mean(df)>=0)
    result={'passed':True,'frozen_files':len(json.loads((RUN/'artifacts/freeze.json').read_text())),'prepared_hash_matches_ancestor':True,'checks':checks,'means':means,'comparison':comp,'future_access':False,'full_training_replayed':False,'audit_script_sha256':sha(RUN/'postrun_audit.py')};save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
