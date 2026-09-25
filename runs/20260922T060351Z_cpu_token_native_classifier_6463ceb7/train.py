import os,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ta_wf_next.traffic_views import generate_views
R=Path(__file__).resolve().parent; ROOT=R.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=10404).reshape(102,102); t=cm.diagonal(); den=cm.sum(0)+cm.sum(1)
 return dict(accuracy=float((y==p).mean()),macro_f1=float(np.divide(2*t,den,out=np.zeros(102),where=den!=0).mean()))
def tokens(rows):
 d=np.zeros((len(rows),512),np.int64); e=np.zeros_like(d,dtype=np.float32); c=np.zeros_like(d)
 for i,row in enumerate(rows):
  v=generate_views(row,input_kind='signed_timestamp',budget=5000,window_sizes=())
  for j,(a,b) in enumerate(zip(v.runs.runs[:512],v.runs.coarse()[:512])):
   d[i,j]=1 if a.direction<0 else 2; e[i,j]=np.log1p(a.count)/np.log(5001); c[i,j]=b.log2_count_bin+1
 return d,e,c
class TokenNet(nn.Module):
 def __init__(self,view):
  super().__init__(); self.view=view
  self.direction=nn.Embedding(3,16,padding_idx=0)
  self.length=nn.Linear(1,16) if view=='exact' else nn.Embedding(14,16,padding_idx=0)
  self.position=nn.Embedding(512,32)
  self.c1=nn.Conv1d(32,32,5,padding=2); self.c2=nn.Conv1d(32,32,5,padding=2); self.head=nn.Linear(512,102)
 def forward(self,d,length):
  mask=(d!=0).unsqueeze(-1)
  le=self.length(length.unsqueeze(-1)) if self.view=='exact' else self.length(length)
  z=(torch.cat([self.direction(d),le],-1)+self.position(torch.arange(512)))*mask
  z=torch.relu(self.c1(z.transpose(1,2))).transpose(1,2)*mask
  z=torch.relu(self.c2(z.transpose(1,2))).transpose(1,2)*mask
  z=z.reshape(-1,16,32,32); m=mask.reshape(-1,16,32,1)
  pooled=z.sum(2)/m.sum(2).clamp_min(1)
  return self.head(pooled.flatten(1))
def main():
 cfg=json.loads((R/'config.json').read_text())
 assert cfg['status']=='frozen' and not cfg['gpu'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert not list((R/'checkpoints').glob('*.pt'))
 torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1)
 dc=json.loads((ROOT/'configs/datasets.json').read_text()); data=Path(dc['data_root'])
 old=ROOT/'runs'/cfg['manifest_source']/'artifacts/manifest.json'; original=json.loads(old.read_text())['sampling']
 arrays={}; ys={}; hs={}
 for role in cfg['roles']:
  item=original[role]
  with np.load(data/item['path'],allow_pickle=False) as f:
   raw=f['X'][item['rows'],:5000]; ys[role]=f['y'][item['rows']].astype(np.int64)
  assert np.isfinite(raw).all() and set(ys[role])==set(range(102))
  hs[role]=[hashlib.sha256(np.sign(a).astype(np.int8).tobytes()).hexdigest() for a in raw]
  values=tokens(raw); assert (values[0]!=0).any(1).all()
  np.savez_compressed(R/'artifacts'/f'{role}_tokens.npz',direction=values[0],exact=values[1],coarse=values[2])
  arrays[role]=tuple(torch.from_numpy(a) for a in values)
 assert not set(hs['source'])&set(hs['valid'])
 (R/'artifacts/manifest.json').write_text(json.dumps(dict(sampling={k:original[k] for k in cfg['roles']},direction_hashes=hs,source_manifest_sha256=sha(old)),indent=2))
 paths=[R/'PLAN.md',R/'config.json',Path(__file__),ROOT/'src/ta_wf_next/traffic_views.py',ROOT/'src/ta_wf_next/burst_tokens.py']
 (R/'artifacts/pretraining_seal.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2))
 y=torch.from_numpy(ys['source']); histories={}; info={}; preds={}
 for seed in cfg['seeds']:
  for view in cfg['views']:
   key=f'{view}_{seed}'; torch.manual_seed(seed); model=TokenNet(view)
   opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
   d=arrays['source'][0]; length=arrays['source'][1 if view=='exact' else 2]
   vd=arrays['valid'][0]; vl=arrays['valid'][1 if view=='exact' else 2]
   generator=torch.Generator().manual_seed(seed); hist=[]; best=-1.; start=time.perf_counter()
   for ep in range(1,cfg['epochs']+1):
    model.train(); total=correct=0
    for ids in torch.randperm(len(y),generator=generator).split(cfg['batch_size']):
     opt.zero_grad(set_to_none=True); q=model(d[ids],length[ids]); loss=nn.functional.cross_entropy(q,y[ids]); assert torch.isfinite(loss)
     loss.backward(); opt.step(); total+=loss.item()*len(ids); correct+=int((q.argmax(1)==y[ids]).sum())
    model.eval()
    with torch.inference_mode(): p=torch.cat([model(vd[i:i+128],vl[i:i+128]).argmax(1) for i in range(0,len(vd),128)]).numpy()
    m=metric(ys['valid'],p); hist.append(dict(epoch=ep,loss=total/len(y),train_accuracy=correct/len(y),**m))
    if m['macro_f1']>best: best=m['macro_f1']; state={k:v.clone() for k,v in model.state_dict().items()}; bestep=ep
   model.load_state_dict(state)
   with torch.inference_mode(): preds[key]=torch.cat([model(vd[i:i+128],vl[i:i+128]).argmax(1) for i in range(0,len(vd),128)]).numpy()
   torch.save(state,R/'checkpoints'/f'{key}.pt')
   reload=TokenNet(view); reload.load_state_dict(torch.load(R/'checkpoints'/f'{key}.pt',weights_only=True)); reload.eval()
   with torch.inference_mode(): torch.testing.assert_close(model(vd[:8],vl[:8]),reload(vd[:8],vl[:8]),rtol=0,atol=0)
   histories[key]=hist; info[key]=dict(best_epoch=bestep,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start)
   (R/'artifacts/history_partial.json').write_text(json.dumps(histories,indent=2)); print(key,info[key],flush=True)
 np.savez_compressed(R/'artifacts/predictions.npz',**preds)
 (R/'artifacts/history.json').write_text(json.dumps(histories,indent=2)); (R/'artifacts/models.json').write_text(json.dumps(info,indent=2))
 files=list((R/'checkpoints').glob('*.pt'))+[R/'artifacts/predictions.npz',R/'artifacts/source_tokens.npz',R/'artifacts/valid_tokens.npz',R/'artifacts/manifest.json']
 (R/'artifacts/output_seal.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
 rows=[dict(view=k,**metric(ys['valid'],p),**info[k]) for k,p in preds.items()]
 (R/'artifacts/metrics.json').write_text(json.dumps(rows,indent=2)); print('COMPLETE',flush=True)
if __name__=='__main__': main()
