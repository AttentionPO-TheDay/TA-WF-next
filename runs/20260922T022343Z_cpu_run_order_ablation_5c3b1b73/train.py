from __future__ import annotations
import csv, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ta_wf_next.traffic_views import generate_views

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
DATA=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
SEED=1729; L=5000; RUN_WIDTH=512; EPOCHS=15; BATCH=128

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)

def select(y, cap):
    rng=np.random.default_rng(SEED)
    return np.concatenate([rng.permutation(np.flatnonzero(y==c))[:cap] for c in range(102)])

def extract(x):
    n=len(x); packet=np.sign(x[:,:L]).astype(np.float32); windows=np.zeros((n,240),np.float32)
    exact=np.zeros((n,RUN_WIDTH,4),np.float32); coarse=np.zeros((n,RUN_WIDTH,8),np.float32); mask=np.zeros((n,RUN_WIDTH),bool)
    for i,row in enumerate(x):
        v=generate_views(row[:L],input_kind='signed_timestamp',budget=L,window_sizes=(50,250))
        pos=0
        for width,entries in v.direction_windows:
            for j,w in enumerate(entries):
                windows[i,pos+2*j:pos+2*j+2]=(w.positive_fraction,w.transition_fraction)
            pos+=2*((L+width-1)//width)
        for j,(r,c) in enumerate(zip(v.runs.runs,v.runs.coarse())):
            if j>=RUN_WIDTH: break
            exact[i,j]=(r.direction,np.log1p(r.count),float(r.touches_left_boundary),float(r.touches_right_boundary))
            coarse[i,j]=(c.direction,c.log2_count_bin,float(c.previous_bin_delta or 0),float(c.previous_bin_delta is not None),float(c.next_bin_delta or 0),float(c.next_bin_delta is not None),float(c.touches_left_boundary),float(c.touches_right_boundary))
            mask[i,j]=True
    return {'packet':packet,'windows':windows,'exact':exact,'coarse':coarse,'mask':mask}

def normalize(data, role='source'):
    out={}
    for key in ('packet','windows','exact','coarse'):
        a=data[key].astype(np.float32,copy=True)
        if key in ('exact','coarse'):
            valid=data['mask']
            mean=a[valid].mean(0); std=a[valid].std(0); std=np.where(std<1e-6,1.,std)
            a=(a-mean)/std; a[~valid]=0
            out[key]=(a,mean.astype(np.float32),std.astype(np.float32))
        else:
            mean=a.mean(0); std=a.std(0); std=np.where(std<1e-6,1.,std)
            out[key]=((a-mean)/std,mean.astype(np.float32),std.astype(np.float32))
    return out

class PacketNet(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Conv1d(1,16,9,stride=4),nn.ReLU(),nn.Conv1d(16,32,7,stride=4),nn.ReLU(),nn.AdaptiveAvgPool1d(32)); self.head=nn.Linear(32*32,102)
    def forward(self,x): return self.head(self.body(x.unsqueeze(1)).flatten(1))

class WindowNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(240,128),nn.ReLU(),nn.Linear(128,102))
    def forward(self,x): return self.net(x)

class RunNet(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.conv1=nn.Conv1d(channels,64,5,padding=2); self.conv2=nn.Conv1d(64,64,5,padding=2); self.head=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,102))
    def forward(self,x,m):
        w=m.unsqueeze(-1); x=x.masked_fill(~w,0); z=torch.relu(self.conv1(x.transpose(1,2))).transpose(1,2)*w; z=torch.relu(self.conv2(z.transpose(1,2))).transpose(1,2)*w; mean=(z*w).sum(1)/w.sum(1).clamp_min(1); neg=z.masked_fill(~m.unsqueeze(-1),-1e9); mx=neg.max(1).values; return self.head(torch.cat([mean,mx],1))

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102); tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((p==y).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den!=0).mean())}

def run_model(name, train_d, valid_d, ytr, yv, key):
    seed_all(); model={'packet':PacketNet,'windows':WindowNet,'exact':lambda:RunNet(4),'coarse':lambda:RunNet(8)}[name](); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
    def tensors(d,y):
        if name in ('exact','coarse'): return TensorDataset(torch.from_numpy(d[name][0]),torch.from_numpy(d['mask']),torch.from_numpy(y))
        return TensorDataset(torch.from_numpy(d[name][0]),torch.from_numpy(y))
    tr=DataLoader(tensors(train_d,ytr),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED)); va=tensors(valid_d,yv); hist=[]; best=None; bestf=-1
    for epoch in range(1,EPOCHS+1):
        model.train(); train_loss=0.; correct=0
        for batch in tr:
            opt.zero_grad(set_to_none=True); logits=model(*batch[:-1]); l=loss_fn(logits,batch[-1]); l.backward(); opt.step()
            train_loss+=l.item()*len(batch[-1]); correct+=int((logits.argmax(1)==batch[-1]).sum())
        model.eval();
        with torch.inference_mode():
            logits=model(*va.tensors[:-1]); pred=logits.argmax(1).numpy()
        m=metric(yv,pred); hist.append({'epoch':epoch,'train_loss':train_loss/len(ytr),'train_accuracy':correct/len(ytr),**m})
        print(key,hist[-1],flush=True)
        if m['macro_f1']>bestf: bestf=m['macro_f1']; best={k:v.detach().clone() for k,v in model.state_dict().items()}; best_epoch=epoch
    model.load_state_dict(best); model.eval()
    torch.save({'state_dict':best,'epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters())},RUN/'checkpoints'/f'{key}.pt')
    return model,hist,best_epoch

def permuted(d, role):
    out={k:v for k,v in d.items()}
    permutations=[]
    rng=np.random.default_rng(9001+['source','valid','jp','subpage'].index(role))
    for m in d['mask']:
        n=int(m.sum()); p=np.arange(RUN_WIDTH); p[:n]=rng.permutation(n); permutations.append(p)
    p=np.stack(permutations)
    for name in ('exact','coarse'):
        a,mean,std=d[name]
        out[name]=(np.take_along_axis(a,p[:,:,None],axis=1),mean,std)
    return out,p

def main():
    global SEED
    config=json.loads((RUN/'config.json').read_text())
    assert config['status']=='frozen' and config['gpu'] is False
    assert not (RUN/'artifacts/predictions.npz').exists(), 'refuse overwrite'
    torch.set_num_interop_threads(1)
    seed_all(); paths={'source':DATA/'NetworkDrift'/'train.npz','valid':DATA/'NetworkDrift'/'valid.npz','jp':DATA/'NetworkDrift'/'JP.npz','subpage':DATA/'BehaviorDrift'/'subpage.npz'}; roles={}; manifest={}
    for role,path in paths.items():
        with np.load(path,allow_pickle=False) as z: x=z['X']; y=z['y'].astype(np.int64)
        cap={'source':20,'valid':5,'jp':20,'subpage':20}[role]; idx=select(y,cap); roles[role]={'features':extract(x[idx]),'y':y[idx], 'idx':idx.tolist()}; manifest[role]={'path':str(path.relative_to(DATA)),'rows':idx.tolist(),'count':len(idx),'label_use':'source fit; valid selection; target scoring only'}; print('features',role,len(idx),flush=True)
        old=json.loads((ROOT/'runs/20260922T011309Z_token_effect_diagnostic_d0b9386f/artifacts/sampling_and_isolation.json').read_text())['manifest'][role]
        assert idx.tolist()==old['rows'] and set(y[idx])==set(range(102))
        assert all(np.isfinite(a).all() for a in roles[role]['features'].values())
        del x
    source=roles['source']['features']
    # Reuse source statistics for all non-source roles.
    stats=normalize(source); normalized={}
    np.savez_compressed(RUN/'artifacts/source_statistics.npz',**{k+'_'+n:s[i] for k,s in stats.items() for n,i in [('mean',1),('std',2)]})
    for role,item in roles.items():
        normalized[role]={}
        for key in ('packet','windows','exact','coarse'):
            a=item['features'][key].astype(np.float32,copy=True); mean,std=stats[key][1],stats[key][2]; a=(a-mean)/std
            if key in ('exact','coarse'): a[~item['features']['mask']]=0
            normalized[role][key]=(a,mean,std)
        normalized[role]['mask']=item['features']['mask']
    shuffled={}; permutations={}
    for role,d in normalized.items(): shuffled[role],permutations[role]=permuted(d,role)
    np.savez_compressed(RUN/'artifacts/permutations.npz',**permutations)
    predictions={}; histories={}; rows=[]; best_epochs={}; keys=[]
    for training_seed in config['training_seeds']:
      SEED=training_seed
      for name in ('exact','coarse'):
       for condition in ('ordered','shuffled'):
        key=f'{name}_{condition}_{training_seed}'; keys.append(key)
        inputs=normalized if condition=='ordered' else shuffled
        model,hist,bepoch=run_model(name,inputs['source'],inputs['valid'],roles['source']['y'],roles['valid']['y'],key)
        histories[key]=hist; best_epochs[key]=bepoch
        for role in ('valid','jp','subpage'):
            d=inputs[role]
            with torch.inference_mode():
                outputs=[]
                for start in range(0,len(d[name][0]),BATCH):
                    xb=torch.from_numpy(d[name][0][start:start+BATCH])
                    logits=model(xb,torch.from_numpy(d['mask'][start:start+BATCH]))
                    outputs.append(logits.argmax(1).numpy())
            predictions[role+'_'+key]=np.concatenate(outputs).astype(np.int16)
        (RUN/'artifacts/history_partial.json').write_text(json.dumps(histories,indent=2))
    np.savez_compressed(RUN/'artifacts/predictions.npz',**predictions); sha=hashlib.sha256((RUN/'artifacts/predictions.npz').read_bytes()).hexdigest()
    (RUN/'artifacts/manifest.json').write_text(json.dumps({'sampling':manifest,'best_epochs':best_epochs,'prediction_sha256':sha},indent=2)); (RUN/'artifacts/history.json').write_text(json.dumps(histories,indent=2))
    seal_paths=[RUN/'artifacts/permutations.npz',RUN/'PLAN.md',RUN/'config.json',Path(__file__),ROOT/'src/ta_wf_next/traffic_views.py',ROOT/'src/ta_wf_next/burst_tokens.py',RUN/'artifacts/source_statistics.npz',*sorted((RUN/'checkpoints').glob('*.pt'))]
    (RUN/'artifacts/seal.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in seal_paths},indent=2))
    for role in ('valid','jp','subpage'):
        for name in keys:
            rows.append({'role':role,'view':name,'best_epoch':best_epochs[name],**metric(roles[role]['y'],predictions[role+'_'+name])})
    with (RUN/'artifacts/metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['role','view','best_epoch','accuracy','macro_f1']); w.writeheader(); w.writerows(rows)
    (RUN/'artifacts/summary.json').write_text(json.dumps({'metrics':rows,'best_epochs':best_epochs,'config':'CPU ordered run CNN; source valid selection; target labels scoring only'},indent=2)); print('COMPLETE')

if __name__=='__main__': main()


