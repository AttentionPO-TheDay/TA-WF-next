import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,numpy as np,torch
from common import RUN,ROOT,config,verify_freeze,deterministic,save,sha,metric
from model import MultiViewModel,span_mask
from worker import predict
def main():
 c=config();verify_freeze();deterministic('cuda',2);raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');data={r:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()};reports=[];checks={'metrics':0,'logits':0,'fixed_mask':0,'initialization':0,'index':0,'selection':0};models={k:MultiViewModel(k).cuda() for k in c['conditions']}
 for s in c['seeds']:
  for k in c['conditions']:
   out=RUN/'artifacts'/f'{k}_{s}';rep=json.loads((out/'report.json').read_text());reports.append(rep);hist=json.loads((out/'history.json').read_text());assert max(hist,key=lambda h:h['valid']['accuracy'])['step']==rep['best_step'];checks['selection']+=1
   ini=torch.load(out/'initial_state.pt',weights_only=True);old=torch.load(ROOT/c['historical_reference_run']/'artifacts'/f'ce_only_{s}'/'initial_state.pt',weights_only=True);assert set(old).issubset(ini) and all(torch.equal(ini[x],old[x]) for x in old);checks['initialization']+=1
   assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True),torch.load(ROOT/c['historical_reference_run']/'artifacts'/f'ce_only_{s}'/'index_stream.pt',weights_only=True));checks['index']+=1
   ck=torch.load(RUN/'checkpoints'/f'{k}_{s}_best.pt',weights_only=True,map_location='cuda');assert ck['config_sha256']==sha(RUN/'config.json');models[k].load_state_dict(ck['state_dict']);z=np.load(out/'logits_best.npz')
   for role,e in data.items():
    assert np.isfinite(z[role]).all();sc=metric(e['labels'].cpu().numpy(),z[role].argmax(1));assert all(np.isclose(sc[m],rep['best'][role][m],atol=1e-12) for m in sc);checks['metrics']+=1;rp=predict(models[k],e,c['eval_batch_size']);assert np.array_equal(rp.argmax(1),z[role].argmax(1));checks['logits']+=1
   ve=data['valid'];fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']));mz=predict(models[k],{**ve,'tam':fixed},c['eval_batch_size']);assert np.isclose(metric(ve['labels'].cpu().numpy(),mz.argmax(1))['accuracy'],rep['fixed_view_diagnostic']['masked_accuracy']);checks['fixed_mask']+=1
 means={k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k])) for m in ('accuracy','macro_f1')} for k in c['conditions']};summary=json.loads((RUN/'artifacts/summary.json').read_text());assert means==summary['means'];a=[r for r in reports if r['kind']=='supcon'];b=[r for r in reports if r['kind']=='ce_control'];da=[100*(x['best']['valid']['accuracy']-y['best']['valid']['accuracy']) for x,y in zip(a,b)];df=[100*(x['best']['valid']['macro_f1']-y['best']['valid']['macro_f1']) for x,y in zip(a,b)];comp=summary['comparison_supcon_minus_ce_control'];assert np.allclose(da,comp['accuracy_per_seed_pp']);result={'passed':True,'frozen_files':len(json.loads((RUN/'artifacts/freeze.json').read_text())),'checks':checks,'means':means,'comparison':comp,'future_access':False,'audit_script_sha256':sha(RUN/'postrun_audit.py')};save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
