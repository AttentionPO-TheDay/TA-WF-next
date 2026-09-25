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
class Branch(nn.Module):
 def __init__(self,kind,seed):
  super().__init__(); self.kind=kind
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed)
   if kind=='run':
    self.d=nn.Embedding(3,16,padding_idx=0); self.b=nn.Embedding(14,16,padding_idx=0)
    self.pos=nn.Embedding(512,32); self.c1=nn.Conv1d(32,32,5,padding=2); self.c2=nn.Conv1d(32,32,5,padding=2)
   else:
    self.w1=nn.Linear(240,32); self.w2=nn.Linear(32,512)
 def forward(self,d,b,m,w,adapter=None):
  if self.kind=='window':
   z=torch.relu(self.w1(w))
   if adapter is not None: z=z+adapter(z)
   return torch.relu(self.w2(z))
  mask=m.unsqueeze(-1)
  z=torch.cat([self.d(d),self.b(b)],-1)
  if adapter is not None: z=z+adapter(z)
  z=(z+self.pos(torch.arange(512)))*mask
  z=torch.relu(self.c1(z.transpose(1,2))).transpose(1,2)*mask
  z=torch.relu(self.c2(z.transpose(1,2))).transpose(1,2)*mask
  return (z.reshape(-1,16,32,32).sum(2)/mask.reshape(-1,16,32,1).sum(2).clamp_min(1)).flatten(1)

class Net(nn.Module):
 def __init__(self,arm,seed):
  super().__init__(); self.arm=arm
  kinds={'run_pair':('run','run'),'window_pair':('window','window')}.get(arm,('run','window'))
  self.left=Branch(kinds[0],seed); self.right=Branch(kinds[1],seed+1000)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed+2000); self.head=nn.Linear(1024,102)
   torch.manual_seed(seed+3000)
   if arm=='fusion_shared': self.shared=self.adapter(16)
   if arm=='fusion_specific':
    self.run_adapter=self.adapter(8); self.window_adapter=self.adapter(8)
 def adapter(self,width):
  a=nn.Sequential(nn.Linear(32,width),nn.GELU(),nn.Linear(width,32))
  nn.init.zeros_(a[2].weight); nn.init.zeros_(a[2].bias); return a
 def forward(self,d,b,m,w):
  a1=a2=None
  if self.arm=='fusion_shared': a1=a2=self.shared
  elif self.arm=='fusion_specific': a1=self.run_adapter; a2=self.window_adapter
  return self.head(torch.cat([self.left(d,b,m,w,a1),self.right(d,b,m,w,a2)],1))

def tests():
 torch.set_num_threads(1)
 d=torch.zeros(2,512,dtype=torch.long); d[:,:12]=2
 b=d.clone(); m=d!=0; w=torch.rand(2,240)
 baseline=Net('fusion',1729); sizes=[]
 for arm in ('run_pair','window_pair','fusion','fusion_shared','fusion_specific'):
  model=Net(arm,1729); sizes.append(sum(p.numel() for p in model.parameters()))
  z=model(d,b,m,w); assert z.shape==(2,102) and torch.isfinite(z).all()
  b2=b.clone(); b2[:,12:]=10; d2=d.clone(); d2[:,12:]=1
  torch.testing.assert_close(z,model(d2,b2,m,w),rtol=0,atol=0)
  if arm=='run_pair': torch.testing.assert_close(z,model(d,b,m,w+2),rtol=0,atol=0)
  if arm=='window_pair': torch.testing.assert_close(z,model(d2,b2,~m,w),rtol=0,atol=0)
  if arm.startswith('fusion'):
   for name,value in baseline.state_dict().items(): torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
   torch.testing.assert_close(z,baseline(d,b,m,w),rtol=0,atol=0)
  if arm in ('fusion_shared','fusion_specific'):
   adapters=[model.shared] if arm=='fusion_shared' else [model.run_adapter,model.window_adapter]
   z.square().sum().backward()
   for a in adapters: assert a[2].weight.grad.abs().sum()>0
   torch.optim.SGD(model.parameters(),lr=.01).step(); model.zero_grad()
   model(d,b,m,w).square().sum().backward()
   for a in adapters: assert a[0].weight.grad.abs().sum()>0
 assert max(sizes)/min(sizes)<1.05
 print('PASS mask, view permissions, shared init, identity output, two-step adapter gradients, capacity',sizes)
def main():
 cfg=json.loads((R/'config.json').read_text()); assert cfg['status']=='frozen' and not cfg['gpu'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert not list((R/'checkpoints').glob('*.pt'))
 torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
 old=ROOT/'runs'/cfg['source_run']; man=json.loads((old/'artifacts/manifest.json').read_text()); data=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
 arrays={}; ys={}; hashes={}; raw_windows={}
 old_seal=json.loads((old/'artifacts/output_seal.json').read_text())
 assert sha(old/'artifacts/manifest.json')==old_seal[str((old/'artifacts/manifest.json').relative_to(ROOT))]
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
  assert hs==man['direction_hashes'][role]
  assert np.isfinite(ww).all() and rm.any(1).all() and set(ys[role])==set(range(102))
  raw_windows[role]=ww.copy()
  hashes[role]=hs; arrays[role]=(torch.from_numpy(rd),torch.from_numpy(rb),torch.from_numpy(rm),torch.from_numpy(ww))
 assert not set(hashes['source'])&set(hashes['valid']); save('manifest.json',{'sampling':man['sampling'],'direction_hashes':hashes})
 mean=raw_windows['source'].mean(0); std=raw_windows['source'].std(0); std=np.where(std<1e-6,1.,std)
 np.savez_compressed(R/'artifacts/window_statistics.npz',mean=mean,std=std)
 for role in cfg['roles']:
  d,b,m,w=arrays[role]; w=torch.from_numpy((raw_windows[role]-mean)/std); arrays[role]=(d,b,m,w)
  np.savez_compressed(R/'artifacts'/f'{role}_inputs.npz',direction=d.numpy(),bucket=b.numpy(),mask=m.numpy(),windows=w.numpy(),raw_windows=raw_windows[role],labels=ys[role])
 save('pretraining_seal.json',{str(p.relative_to(ROOT)):sha(p) for p in [R/'PLAN.md',R/'config.json',Path(__file__),R/'verify.py',R/'artifacts/source_inputs.npz',R/'artifacts/valid_inputs.npz',R/'artifacts/window_statistics.npz',ROOT/'configs/datasets.json',ROOT/'src/ta_wf_next/traffic_views.py',ROOT/'src/ta_wf_next/burst_tokens.py',old/'artifacts/manifest.json']})
 y=torch.from_numpy(ys['source']); histories={}; rows=[]; preds={}; info={}
 for seed in cfg['seeds']:
  for arm in cfg['arms']:
   key=f'{arm}_{seed}'; started=time.perf_counter(); torch.manual_seed(seed); model=Net(arm,seed); opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); gen=torch.Generator().manual_seed(seed); best=-1; hist=[]
   for ep in range(1,cfg['epochs']+1):
    model.train(); total=correct=0
    for ids in torch.randperm(len(y),generator=gen).split(cfg['batch_size']):
     opt.zero_grad(set_to_none=True); q=model(*(x[ids] for x in arrays['source'])); loss=nn.functional.cross_entropy(q,y[ids]); assert torch.isfinite(loss); loss.backward(); opt.step(); total+=loss.item()*len(ids); correct+=int((q.argmax(1)==y[ids]).sum())
    model.eval();
    with torch.inference_mode(): p=torch.cat([model(*(x[i:i+128] for x in arrays['valid'])).argmax(1) for i in range(0,len(ys['valid']),128)]).numpy()
    mm=metric(ys['valid'],p); hist.append({'epoch':ep,'loss':total/len(y),'train_accuracy':correct/len(y),**mm})
    if mm['macro_f1']>best: best=mm['macro_f1']; state={k:v.clone() for k,v in model.state_dict().items()}; be=ep; bp=p.copy()
    if ep in (15,30,45):
     np.savez_compressed(R/'artifacts'/f'{key}_budget{ep}.npz',selected=bp,final=p)
     print(key,'epoch',ep,'train_acc',hist[-1]['train_accuracy'],'valid',mm,'selected',be,flush=True)
   model.load_state_dict(state); torch.save(state,R/'checkpoints'/f'{key}.pt'); reload=Net(arm,seed); reload.load_state_dict(torch.load(R/'checkpoints'/f'{key}.pt',weights_only=True)); reload.eval()
   with torch.inference_mode(): rp=torch.cat([reload(*(x[i:i+128] for x in arrays['valid'])).argmax(1) for i in range(0,len(ys['valid']),128)]).numpy()
   np.testing.assert_array_equal(rp,bp); preds[key]=rp; histories[key]=hist; info[key]={'best_epoch':be,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started}; rows.append({'key':key,'arm':arm,'seed':seed,**metric(ys['valid'],rp),**info[key]}); save('history_partial.json',histories); print(rows[-1],flush=True)
 np.savez_compressed(R/'artifacts/predictions.npz',**preds); save('history.json',histories); save('metrics.json',rows); save('output_seal.json',{str(p.relative_to(ROOT)):sha(p) for p in list((R/'checkpoints').glob('*.pt'))+list((R/'artifacts').glob('*.npz'))+[R/'artifacts/metrics.json',R/'artifacts/history.json']}); print('COMPLETE')
if __name__=='__main__':
 import sys
 tests() if '--test' in sys.argv else main()
