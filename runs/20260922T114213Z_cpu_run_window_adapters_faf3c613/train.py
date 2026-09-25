import hashlib,json,os,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from ta_wf_next.traffic_views import generate_views
R=Path(__file__).resolve().parent; ROOT=R.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=10404).reshape(102,102); den=cm.sum(0)+cm.sum(1)
 return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den!=0).mean())}
def save(n,x): (R/'artifacts'/n).write_text(json.dumps(x,indent=2))
class Net(nn.Module):
 def __init__(self,arm,seed):
  super().__init__(); self.arm=arm
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed); self.d=nn.Embedding(3,16,padding_idx=0); self.b=nn.Embedding(14,16,padding_idx=0); self.pos=nn.Embedding(512,32); self.c1=nn.Conv1d(32,32,5,padding=2); self.c2=nn.Conv1d(32,32,5,padding=2)
   torch.manual_seed(seed+10); self.w=nn.Sequential(nn.Linear(240,64),nn.GELU(),nn.Linear(64,128)); self.rhead=nn.Linear(128,102); self.whead=nn.Linear(128,102); self.fhead=nn.Linear(256,102)
   torch.manual_seed(seed+20); self.ra=nn.Sequential(nn.Linear(32,8),nn.GELU(),nn.Linear(8,32)); self.wa=nn.Sequential(nn.Linear(128,8),nn.GELU(),nn.Linear(8,128))
   nn.init.zeros_(self.ra[2].weight); nn.init.zeros_(self.ra[2].bias); nn.init.zeros_(self.wa[2].weight); nn.init.zeros_(self.wa[2].bias)
 def run(self,d,b,m):
  z=torch.cat([self.d(d),self.b(b)],-1)+self.pos(torch.arange(512)); z=z*m.unsqueeze(-1); z=z+self.ra(z)
  z=torch.relu(self.c1(z.transpose(1,2))).transpose(1,2)*m.unsqueeze(-1); z=torch.relu(self.c2(z.transpose(1,2))).transpose(1,2)*m.unsqueeze(-1)
  q=z.reshape(-1,16,32,32); mm=m.reshape(-1,16,32,1); return (q.sum(2)/mm.sum(2).clamp_min(1)).flatten(1)
 def forward(self,d,b,m,w):
  r=self.run(d,b,m); q=self.w(w); q=q+self.wa(q)
  if self.arm=='run_adapter': return self.rhead(r)
  if self.arm=='window_adapter': return self.whead(q)
  return self.fhead(torch.cat([r,q],1))
def main():
 cfg=json.loads((R/'config.json').read_text()); assert cfg['status']=='frozen' and not cfg['gpu'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
 old=ROOT/'runs'/cfg['source_run']; man=json.loads((old/'artifacts/manifest.json').read_text()); data=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
 arrays={}; ys={}; hashes={}
 for role in cfg['roles']:
  item=man['sampling'][role]
  with np.load(data/item['path'],allow_pickle=False) as z: raw=z['X'][item['rows'],:5000]; ys[role]=z['y'][item['rows']].astype(np.int64)
  hs=[]; rd=np.zeros((len(raw),512),np.int64); rb=np.zeros_like(rd); rm=np.zeros_like(rd,bool); ww=np.zeros((len(raw),240),np.float32)
  for i,row in enumerate(raw):
   hs.append(hashlib.sha256(np.sign(row).astype(np.int8).tobytes()).hexdigest()); v=generate_views(row,input_kind='signed_timestamp',budget=5000,window_sizes=(50,250)); runs=v.runs.runs[:512]
   for j,x in enumerate(runs): rd[i,j]=1 if x.direction<0 else 2; rb[i,j]=x.count.bit_length(); rm[i,j]=1
   pos=0
   for width,entries in v.direction_windows:
    for j,x in enumerate(entries): ww[i,pos+2*j:pos+2*j+2]=(x.positive_fraction,x.transition_fraction)
    pos+=2*((5000+width-1)//width)
  hashes[role]=hs; arrays[role]=(torch.from_numpy(rd),torch.from_numpy(rb),torch.from_numpy(rm),torch.from_numpy(ww))
 assert not set(hashes['source'])&set(hashes['valid']); save('manifest.json',{'sampling':man['sampling'],'direction_hashes':hashes})
 save('pretraining_seal.json',{str(p.relative_to(ROOT)):sha(p) for p in [R/'PLAN.md',R/'config.json',Path(__file__),ROOT/'src/ta_wf_next/traffic_views.py',ROOT/'src/ta_wf_next/burst_tokens.py',old/'artifacts/manifest.json']})
 y=torch.from_numpy(ys['source']); histories={}; rows=[]; preds={}; info={}
 for seed in cfg['seeds']:
  for arm in cfg['arms']:
   key=f'{arm}_{seed}'; torch.manual_seed(seed); model=Net(arm,seed); opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); gen=torch.Generator().manual_seed(seed); best=-1; hist=[]
   for ep in range(1,cfg['epochs']+1):
    model.train(); total=correct=0
    for ids in torch.randperm(len(y),generator=gen).split(cfg['batch_size']):
     opt.zero_grad(set_to_none=True); q=model(*(x[ids] for x in arrays['source'])); loss=nn.functional.cross_entropy(q,y[ids]); loss.backward(); opt.step(); total+=loss.item()*len(ids); correct+=int((q.argmax(1)==y[ids]).sum())
    model.eval();
    with torch.inference_mode(): p=torch.cat([model(*(x[i:i+128] for x in arrays['valid'])).argmax(1) for i in range(0,len(ys['valid']),128)]).numpy()
    mm=metric(ys['valid'],p); hist.append({'epoch':ep,'loss':total/len(y),'train_accuracy':correct/len(y),**mm})
    if mm['macro_f1']>best: best=mm['macro_f1']; state={k:v.clone() for k,v in model.state_dict().items()}; be=ep; bp=p.copy()
   model.load_state_dict(state); torch.save(state,R/'checkpoints'/f'{key}.pt'); reload=Net(arm,seed); reload.load_state_dict(torch.load(R/'checkpoints'/f'{key}.pt',weights_only=True)); reload.eval()
   with torch.inference_mode(): rp=torch.cat([reload(*(x[i:i+128] for x in arrays['valid'])).argmax(1) for i in range(0,len(ys['valid']),128)]).numpy()
   np.testing.assert_array_equal(rp,bp); preds[key]=rp; histories[key]=hist; info[key]={'best_epoch':be,'parameters':sum(p.numel() for p in model.parameters())}; rows.append({'key':key,'arm':arm,'seed':seed,**metric(ys['valid'],rp),**info[key]}); save('history_partial.json',histories); print(rows[-1],flush=True)
 np.savez_compressed(R/'artifacts/predictions.npz',**preds); save('history.json',histories); save('metrics.json',rows); save('output_seal.json',{str(p.relative_to(ROOT)):sha(p) for p in list((R/'checkpoints').glob('*.pt'))+list((R/'artifacts').glob('*.npz'))+[R/'artifacts/metrics.json',R/'artifacts/history.json']}); print('COMPLETE')
if __name__=='__main__': main()
