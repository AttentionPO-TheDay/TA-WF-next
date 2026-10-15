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
    assert sha(RUN/'artifacts/prepared.pt')=='49b01254108433528398d17fd5a4df9618a7632ba03c0a63864236786e9fc4db'
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert set(raw)=={'source','valid'}
    data={r:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}
    summary=json.loads((RUN/'artifacts/summary.json').read_text());reports=[];checks={'metrics':0,'logits_and_checkpoints':0,'fixed_mask':0,'initialization':0,'index_streams':0,'selection':0}
    model=MultiViewModel('ce_only').cuda()
    for seed in c['seeds']:
        out=RUN/'artifacts'/f'ce_trajectory_{seed}';report=json.loads((out/'report.json').read_text());reports.append(report)
        assert report['steps_completed']==12800 and not report['future_access'] and report['swa_steps']==c['swa_steps']
        history=json.loads((out/'history.json').read_text());assert [h['step'] for h in history]==c['eval_steps']
        best=max(history,key=lambda h:h['valid']['accuracy']);assert best['step']==report['best_step'];checks['selection']+=1
        old=ROOT/c['historical_reference_run']/'artifacts'/f'ce_only_{seed}'
        ini=torch.load(out/'initial_state.pt',weights_only=True);ref=torch.load(old/'initial_state.pt',weights_only=True)
        assert ini.keys()==ref.keys() and all(torch.equal(ini[k],ref[k]) for k in ini);checks['initialization']+=1
        assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True),torch.load(old/'index_stream.pt',weights_only=True));checks['index_streams']+=1
        for kind in ['ce_final','swa']:
            path=RUN/'checkpoints'/f'{kind}_{seed}_{"best" if kind=="ce_final" else "fixed"}.pt'
            ck=torch.load(path,weights_only=True,map_location='cuda');assert ck['config_sha256']==sha(RUN/'config.json')
            assert (ck['step']==best['step']) if kind=='ce_final' else (ck['steps']==c['swa_steps'])
            model.load_state_dict(ck['state_dict']);logits=np.load(out/f'{kind}_logits.npz')
            for role,e in data.items():
                z=logits[role];assert z.shape==(len(e['labels']),102) and np.isfinite(z).all()
                score=metric(e['labels'].cpu().numpy(),z.argmax(1))
                for key,value in score.items():assert np.isclose(value,report[kind][role][key],atol=1e-12)
                checks['metrics']+=1
                replay=predict(model,e,c['eval_batch_size']);assert np.allclose(replay,z,atol=1e-5,rtol=1e-5) and np.array_equal(replay.argmax(1),z.argmax(1));checks['logits_and_checkpoints']+=1
            ve=data['valid'];fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']));mz=predict(model,{**ve,'tam':fixed},c['eval_batch_size']);score=metric(ve['labels'].cpu().numpy(),mz.argmax(1));assert np.isclose(score['accuracy'],report['fixed_view_diagnostic'][kind]['masked_accuracy']);checks['fixed_mask']+=1
        print(json.dumps({'audited_seed':seed}),flush=True)
    means={k:{m:float(np.mean([r[k]['valid'][m] for r in reports])) for m in ('accuracy','macro_f1')} for k in ['ce_final','swa']}
    assert means==summary['means']
    da=[100*(r['swa']['valid']['accuracy']-r['ce_final']['valid']['accuracy']) for r in reports]
    df=[100*(r['swa']['valid']['macro_f1']-r['ce_final']['valid']['macro_f1']) for r in reports]
    comp=summary['comparison_swa_minus_ce_final'];assert np.allclose(da,comp['accuracy_per_seed_pp']);assert np.isclose(np.mean(da),comp['accuracy_mean_delta_pp']);assert np.isclose(np.mean(df),comp['macro_f1_mean_delta_pp']);assert comp['positive_seed_count']==sum(v>0 for v in da);assert comp['pass']==bool(np.mean(da)>=1 and min(da)>0 and np.mean(df)>=0)
    result={'passed':True,'frozen_files':len(json.loads((RUN/'artifacts/freeze.json').read_text())),'prepared_hash_matches_ancestor':True,'checks':checks,'means':means,'comparison':comp,'future_access':False,'full_training_replayed':False,'swa_average_reconstructed':False,'limitation':'Seven individual snapshots were not persisted, so the exact SWA average cannot be independently reconstructed; no BN recalibration and no SWA LR exploration in this run.','audit_script_sha256':sha(RUN/'postrun_audit.py')}
    save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
