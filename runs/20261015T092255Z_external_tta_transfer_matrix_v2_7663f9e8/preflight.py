import json,os,hashlib,numpy as np,torch
import port_base as B
from external_data import wtt_dataset,awf_dataset
from model import MultiViewModel
import runner as Q
R,ROOT,C,DEVICE=B.R,B.ROOT,B.C,B.DEVICE
Q.setup(); assert C['status']=='frozen' and os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu'])
summary={}
for name,fn in [('wtt_time',wtt_dataset),('awf',awf_dataset)]:
 source,nclasses,targets=fn(); assert len(source)>0 and nclasses >= 100 and targets
 x,y,u=source[0]; assert x.shape==(5000,) and np.isfinite(x).all()
 model=MultiViewModel('wide_bn',nclasses).to(DEVICE); model.eval();
 with torch.no_grad(): z,_=Q.forward(model,torch.from_numpy(np.stack([x]))); assert z.shape==(1,nclasses) and torch.isfinite(z).all()
 summary[name]={'source_rows':len(source),'classes':nclasses,'targets':{k:{'reference_rows':len(v['reference']),'query_rows':len(v['query']),'phase':v['phase']} for k,v in targets.items()}}
B.save_json('preflight.json',{'passed':True,'summary':summary,'query_adaptation':False,'direction_frontend':True,'config_sha':B.sha(R/'config.json'),'code_sha':{p.name:B.sha(p) for p in [R/'runner.py',R/'external_data.py',R/'port_base.py',R/'model.py']}})
print(json.dumps(summary,ensure_ascii=False),flush=True)
