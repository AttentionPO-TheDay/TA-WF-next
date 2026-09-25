from __future__ import annotations
import csv, hashlib, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ta_wf_next.traffic_views import generate_views

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]; L=5000; BATCH=128; EPOCHS=15
MANIFEST=ROOT/'runs/20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37/artifacts/manifest.json'
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.set_num_threads(4)
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102); tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
 return {'accuracy':float((p==y).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den!=0).mean())}
def extract(x,neutral=False):
 out=np.zeros((len(x),240),np.float32); partial=np.zeros((len(x),120),bool)
 for i,row in enumerate(x):
  v=generate_views(row[:L],input_kind='signed_timestamp',budget=L,window_sizes=(50,250)); pos=0; q=0
  for width,entries in v.direction_windows:
   for j,w in enumerate(entries):
    partial[i,q]=w.partial
    out[i,pos+2*j:pos+2*j+2]=((0.5,0.) if neutral and w.partial else (w.positive_fraction,w.transition_fraction)); q+=1
   pos+=2*((L+width-1)//width)
 return out,partial
class Net(nn.Module):
 def __init__(self,adapter=False):
  super().__init__(); self.adapter=nn.Sequential(nn.Linear(240,32),nn.ReLU(),nn.Linear(32,240)) if adapter else None
  if self.adapter: nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
  self.body=nn.Sequential(nn.Linear(240,128),nn.ReLU(),nn.Linear(128,102))
 def forward(self,x): return self.body(x+(self.adapter(x) if self.adapter else 0))
def main():
 cfg=json.loads((RUN/'config.json').read_text()); assert cfg['status']=='frozen' and not cfg['gpu']
 torch.set_num_interop_threads(1); base=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])/'TemporalDrift'; old=json.loads(MANIFEST.read_text())['sampling']
 roles={}; stats=None; partial_stats={}
 for role in ['source','valid','day14','day30','day90','day150','day270']:
  path=base/('train.npz' if role=='source' else ('valid.npz' if role=='valid' else role.replace('day','day')+'.npz'))
  with np.load(path,allow_pickle=False) as z: x,y=z['X'],z['y'].astype(np.int64)
  idx=np.asarray(old[role]['rows']); a,pm=extract(x[idx],neutral=False); an,_=extract(x[idx],neutral=True); roles[role]={'y':y[idx],'full':a,'neutral':an}; partial_stats[role]={'partial_windows':int(pm.sum()),'total_windows':int(pm.size)}
  if role=='source': stats=(a.mean(0),np.where(a.std(0)<1e-6,1.,a.std(0)),an.mean(0),np.where(an.std(0)<1e-6,1.,an.std(0)))
  del x
 for role,d in roles.items():
  d['full']=(d['full']-stats[0])/stats[1]; d['neutral']=(d['neutral']-stats[2])/stats[3]
 predictions={}; rows=[]; histories={}; best_epochs={}
 for s in cfg['training_seeds']:
  for cond in ['full','adapter','neutral']:
   seed_all(s); key=f'{cond}_{s}'; model=Net(cond=='adapter'); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss=nn.CrossEntropyLoss(); xtr=roles['source']['neutral'] if cond=='neutral' else roles['source']['full']; xva=roles['valid']['full'] if cond!='neutral' else roles['valid']['neutral']; tr=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(roles['source']['y'])),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(s)); bestf=-1; hist=[]
   for ep in range(1,EPOCHS+1):
    model.train(); correct=0; total=0; tl=0
    for xb,yb in tr:
     if cond=='neutral': xb=xb # source neutral is substituted below
     opt.zero_grad(); z=model(xb); l=loss(z,yb); l.backward(); opt.step(); tl+=l.item()*len(yb); correct+=int((z.argmax(1)==yb).sum()); total+=len(yb)
    model.eval();
    with torch.inference_mode(): p=model(torch.from_numpy(xva)).argmax(1).numpy()
    m=metric(roles['valid']['y'],p); rec={'epoch':ep,'train_loss':tl/total,'train_accuracy':correct/total,**m}; hist.append(rec)
    if m['macro_f1']>bestf: bestf=m['macro_f1']; best={k:v.detach().clone() for k,v in model.state_dict().items()}; be=ep
   model.load_state_dict(best); best_epochs[key]=be; histories[key]=hist; torch.save({'state_dict':best,'epoch':be},RUN/'checkpoints'/f'{key}.pt')
   for role,d in roles.items():
    xx=d['full'] if cond!='neutral' else d['neutral'];
    with torch.inference_mode(): pred=model(torch.from_numpy(xx)).argmax(1).numpy()
    predictions[f'{role}_{key}']=pred.astype(np.int16); rows.append({'role':role,'condition':cond,'seed':s,'best_epoch':be,**metric(d['y'],pred)})
 np.savez_compressed(RUN/'artifacts/predictions.npz',**predictions); (RUN/'artifacts/history.json').write_text(json.dumps(histories,indent=2)); (RUN/'artifacts/metrics.csv').write_text('role,condition,seed,best_epoch,accuracy,macro_f1\n'+'\n'.join(','.join(str(r[k]) for k in ['role','condition','seed','best_epoch','accuracy','macro_f1']) for r in rows)+'\n'); (RUN/'artifacts/manifest.json').write_text(json.dumps({'sampling_manifest':str(MANIFEST.relative_to(ROOT)),'partial_stats':partial_stats,'best_epochs':best_epochs,'prediction_sha256':hashlib.sha256((RUN/'artifacts/predictions.npz').read_bytes()).hexdigest()},indent=2)); print('COMPLETE')
if __name__=='__main__': main()
