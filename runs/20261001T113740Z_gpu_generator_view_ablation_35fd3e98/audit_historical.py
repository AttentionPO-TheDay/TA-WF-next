"""Explicit historical fusion audit; these checkpoints never initialize training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import json,hashlib,shutil,gc
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from model import MultiViewModel
R=Path(__file__).resolve().parent;ROOT=R.parents[1]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def main():
 c=json.loads((R/'config.json').read_text());old=ROOT/c['historical_run']
 assert c['roles']==['source','valid'] and not c['future_access']
 for path,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/path)==h,path
 assert sha(R/'base_model.py')==sha(old/'model.py')
 assert sha(R/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 raw=torch.load(R/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
 data={role:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for role,e in raw.items()}
 reports=[];verified=0;frozen_inputs={}
 for seed in c['seeds']:
  source=old/'artifacts'/f'learned_transformer_{seed}'
  target=R/'artifacts'/f'fusion_{seed}';target.mkdir(exist_ok=False)
  origin=json.loads((source/'report.json').read_text())
  assert origin['steps_completed']==c['optimizer_steps'] and origin['validation_opportunities']==len(c['eval_steps'])
  torch.manual_seed(seed);m=MultiViewModel('fusion').cuda()
  initial=torch.load(source/'initial_state.pt',weights_only=True)
  assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
  checkpoint=old/'checkpoints'/f'learned_transformer_{seed}_best.pt'
  m.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict']);m.eval()
  with np.load(source/'predictions_best.npz') as saved:
   for role,e in data.items():
    chunks=[]
    with torch.inference_mode():
     for i in range(0,len(e['labels']),c['eval_batch_size']):
      chunks.append(m(e['timestamps'][i:i+64],e['tam'][i:i+64]).cpu().numpy())
    pred=np.concatenate(chunks).argmax(1);assert np.array_equal(pred,saved[role]),(seed,role)
    verified+=1
  for filename in ('predictions_best.npz','predictions_last.npz','initial_state.pt','index_stream.pt','logits_best.npz'):
   shutil.copyfile(source/filename,target/filename)
   frozen_inputs[str((source/filename).relative_to(ROOT))]=sha(source/filename)
   frozen_inputs[str((target/filename).relative_to(ROOT))]=sha(target/filename)
  for ck in ('best','last'):
   with np.load(target/f'predictions_{ck}.npz') as pred:
    for role in c['roles']:
     y=raw[role]['labels'].numpy();p=pred[role]
     assert abs(accuracy_score(y,p)-origin[ck][role]['accuracy'])<1e-12
     assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-origin[ck][role]['macro_f1'])<1e-12
  report={**origin,'task':f'fusion_{seed}','kind':'fusion','historical_reuse':True,'origin_run':old.name,'origin_task':origin['task'],'reloaded_with_new_fusion_code':True}
  (target/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  frozen_inputs[str((source/'report.json').relative_to(ROOT))]=sha(source/'report.json')
  frozen_inputs[str((target/'report.json').relative_to(ROOT))]=sha(target/'report.json')
  frozen_inputs[str(checkpoint.relative_to(ROOT))]=sha(checkpoint)
  reports.append(report)
  del m,initial;gc.collect();torch.cuda.empty_cache()
 out={'historical_reuse':True,'roles':c['roles'],'future_access':False,'optimizer_steps':0,
      'fusion_reloaded_predictions':verified,'best_last_metric_checks':12,'matching_initialization':'passed',
      'original_freeze_and_data_hashes':'passed','reports':reports,'frozen_inputs':frozen_inputs}
 (R/'artifacts/historical_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('reports','frozen_inputs')},indent=2),flush=True)

if __name__=='__main__':main()
