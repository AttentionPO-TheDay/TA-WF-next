import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; rows=list(csv.DictReader((R/'artifacts/metrics.csv').open())); q={(r['role'],r['view']):r for r in rows}; out=[]
roles=['valid','day14','day30','day90','day150','day270']; views=['packet','windows','exact','coarse']; seeds=[1729,3407,2026]
for view in views:
 valid=np.array([float(q['valid',f'{view}_{s}']['accuracy'])*100 for s in seeds])
 for role in roles:
  rec={'view':view,'role':role}
  for metric in ['accuracy','macro_f1']:
   a=np.array([float(q[role,f'{view}_{s}'][metric])*100 for s in seeds]); rec[metric]={'values':a.tolist(),'mean':a.mean(),'sd':a.std(ddof=1)}
  rec['accuracy']['drop_from_valid']=(valid-np.array(rec['accuracy']['values'])).tolist()
  rec['accuracy']['drop_mean']=np.mean(rec['accuracy']['drop_from_valid']); out.append(rec)
positive={}
for view in ['windows','exact','coarse']:
 dates=[]
 for role in roles[1:]:
  a=next(x for x in out if x['view']==view and x['role']==role)['accuracy']['mean']; b=next(x for x in out if x['view']=='packet' and x['role']==role)['accuracy']['mean']; dates.append({'role':role,'delta_vs_packet':a-b})
 positive[view]={'dates':dates,'count_positive':sum(x['delta_vs_packet']>0 for x in dates),'candidate_positive':sum(x['delta_vs_packet']>0 for x in dates)>=4}
(R/'artifacts/date_summary.json').write_text(json.dumps({'rows':out,'comparison':positive},indent=2))
for role in roles:
 print(role,[(v,round(next(x for x in out if x['view']==v and x['role']==role)['accuracy']['mean'],2)) for v in views])
print(positive)
