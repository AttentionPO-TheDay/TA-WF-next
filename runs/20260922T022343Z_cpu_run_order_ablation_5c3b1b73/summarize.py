import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'artifacts/metrics.csv').open()))
lookup={(r['role'],r['view']):r for r in rows}
out=[]
for view in ['exact','coarse']:
 for role in ['valid','jp','subpage']:
  record={'view':view,'role':role}
  for metric in ['accuracy','macro_f1']:
   a=np.array([float(lookup[role,f'{view}_ordered_{s}'][metric])*100 for s in [1729,3407,2026]])
   b=np.array([float(lookup[role,f'{view}_shuffled_{s}'][metric])*100 for s in [1729,3407,2026]])
   record[metric]={'ordered':a.tolist(),'shuffled':b.tolist(),'ordered_mean':a.mean(),'shuffled_mean':b.mean(),'ordered_sd':a.std(ddof=1),'shuffled_sd':b.std(ddof=1),'paired_delta':(a-b).tolist(),'delta_mean':(a-b).mean(),'delta_sd':(a-b).std(ddof=1)}
  out.append(record)
decision={}
for view in ['exact','coarse']:
 decision[view]=all(all(d>0 for d in r['accuracy']['paired_delta']) for r in out if r['view']==view and r['role'] in ['valid','jp'])
(R/'artifacts/paired_summary.json').write_text(json.dumps({'rows':out,'consistent_valid_and_jp':decision},indent=2))
for r in out:
 v=r['accuracy']; print(r['view'],r['role'], 'ordered %.2f shuffled %.2f delta %.2f'% (v['ordered_mean'],v['shuffled_mean'],v['delta_mean']),v['paired_delta'])
print('consistent',decision)
