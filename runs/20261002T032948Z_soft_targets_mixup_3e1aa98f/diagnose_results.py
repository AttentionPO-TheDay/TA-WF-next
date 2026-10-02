"""Post-run descriptive diagnosis of already evaluated source/valid only."""
import json,numpy as np,torch
from common import RUN,ROOT,save

def main():
 torch.set_num_threads(2);c=json.loads((RUN/'config.json').read_text());raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');y=raw['valid']['labels'].numpy();out={'marginal_target':[],'error_overlap':{},'last_accuracy_means':{},'scope':'descriptive development diagnosis, no oracle route/ensemble trained or scored'}
 preds={}
 for k in c['reported_conditions']:
  preds[k]=[];last=[]
  for s in c['seeds']:
   with np.load(RUN/'artifacts'/f'{k}_{s}'/'predictions_best.npz') as f:preds[k].append(f['valid'].copy())
   r=json.loads((RUN/'artifacts'/f'{k}_{s}'/'report.json').read_text());last.append(r['last']['valid']['accuracy'])
  out['last_accuracy_means'][k]=float(np.mean(last))
 rf=ROOT/'runs/20261001T091714Z_gpu_native_varcnn_rf_source_ce09ee6b';rfpred=[]
 for s in c['seeds']:
  t=torch.load(RUN/'artifacts'/f'teacher_source_{s}.pt',weights_only=True);q=(t['source_logits']/2).softmax(1).mean(0)
  out['marginal_target'].append({'seed':s,'kl_to_uniform':float((q*(q.log()+np.log(102))).sum()),'min_probability':float(q.min()),'max_probability':float(q.max()),'uniform_probability':1/102})
  with np.load(rf/'artifacts'/f'rf_{s}'/'predictions_best.npz') as f:rfpred.append(f['valid'].copy())
 rfc=np.array(rfpred)==y
 for k,p in preds.items():
  correct=np.array(p)==y;wrong=~correct.any(0)
  out['error_overlap'][k]={'common_wrong_all_3_seeds':int(wrong.sum()),'among_common_wrong_rf_all_3_correct':int((wrong&rfc.all(0)).sum()),'among_common_wrong_rf_at_least_2_correct':int((wrong&(rfc.sum(0)>=2)).sum())}
 save(RUN/'artifacts/diagnosis.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
