from __future__ import annotations
import argparse, csv, hashlib, json, re, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ta_wf_next.traffic_views import generate_views
DATA_CONFIG=json.loads((ROOT/'configs/datasets.json').read_text())
DATA_ROOT=Path(DATA_CONFIG['data_root'])
PCAP_ROOT=DATA_ROOT/DATA_CONFIG['datasets']['pcap_unlabeled']['path']
PRIOR = ROOT / 'runs/20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37/artifacts/manifest.json'
NPZ_ROOT=DATA_ROOT/DATA_CONFIG['datasets']['proteus_temporal']['path']
SEEDS = [1729, 3407, 2026]
WIDTHS = (50, 250)
PACKET_BUDGET = 5000
WINDOW_DIM = 240
DEVICE_BY_CONDITION = {'scratch_fusion': 'cuda:0', 'pcap_packet_pretrain': 'cuda:1', 'pcap_window_pretrain': 'cuda:2'}

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102); tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den!=0).mean())}

def parse_pcap(path, cap):
    # tcpdump is used only as a read-only decoder; labels and file names never enter features.
    rgx=re.compile(r'^\s*([0-9.]+)\s+IP6?\s+(.+?) > (.+?):\s+([A-Za-z]+)')
    try: out=subprocess.run(['tcpdump','-tt','-nn','-q','-r',str(path),'-c',str(cap)],capture_output=True,text=True,timeout=180)
    except Exception: return []
    flows={}
    for line in out.stdout.splitlines():
        m=rgx.match(line)
        if not m: continue
        t,src,dst,proto=m.groups(); proto=proto.lower()
        key=(proto,tuple(sorted((src,dst))))
        item=flows.setdefault(key,{'first':src,'rows':[]}); item['rows'].append((float(t),src))
    if not flows: return []
    best=max(flows.values(),key=lambda x:len(x['rows']))
    seq=[1 if src==best['first'] else -1 for _,src in best['rows']]
    chunks=[]
    for start in range(0,len(seq),PACKET_BUDGET):
        part=seq[start:start+PACKET_BUDGET]
        if len(part)<1000: continue
        a=np.zeros(PACKET_BUDGET,dtype=np.float32); a[:len(part)]=part; chunks.append(a)
    return chunks

def prepare():
    out=RUN/'artifacts/pcap_sequences.npz'
    if out.exists(): return
    folders=sorted(p for p in PCAP_ROOT.iterdir() if p.is_dir())
    groups=[sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.pcap','.pcapng')) for folder in folders]
    files=[p for j in range(max(map(len,groups))) for group in groups for p in group[j:j+1]]
    seq=[]; file_info=[]; started=time.monotonic()
    for i,p in enumerate(files):
        got=parse_pcap(p,30000); seq.extend(got); file_info.append({'path':str(p),'suffix':p.suffix,'size':p.stat().st_size,'sequences':len(got)})
        if len(seq)>=512: break
        if (i+1)%10==0: print('pcap',i+1,'sequences',len(seq),flush=True)
    seq=seq[:512]
    if len(seq)<32: raise RuntimeError(f'only {len(seq)} usable PCAP sequences')
    np.savez_compressed(out, packets=np.stack(seq))
    (RUN/'artifacts/pcap_manifest.json').write_text(json.dumps({'files_considered':file_info,'sequence_count':len(seq),'sha256':sha(out),'elapsed_seconds':time.monotonic()-started},indent=2))

def windows(rows):
    out=np.zeros((len(rows),WINDOW_DIM),dtype=np.float32)
    for i,row in enumerate(rows):
        off=0
        views=generate_views(row[:PACKET_BUDGET],input_kind='signed_timestamp',budget=PACKET_BUDGET,window_sizes=WIDTHS)
        for width,items in views.direction_windows:
            for j,item in enumerate(items):
                out[i,off+2*j:off+2*j+2]=(0.5,0.0) if item.partial else (item.positive_fraction,item.transition_fraction)
            off+=2*((PACKET_BUDGET+width-1)//width)
    return out

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.packet_body=nn.Sequential(nn.Conv1d(1,32,9,stride=4),nn.ReLU(),nn.Conv1d(32,64,7,stride=4),nn.ReLU(),nn.AdaptiveAvgPool1d(16),nn.Flatten(),nn.Linear(1024,128),nn.ReLU())
        self.window_body=nn.Sequential(nn.Linear(WINDOW_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.head=nn.Linear(256,102)
    def forward(self,p,w): return self.head(torch.cat((self.packet_body(p.unsqueeze(1)),self.window_body(w)),1))

def seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def pretrain_body(kind, ext, epochs, device, seed_value):
    seed(seed_value); body=Fusion().to(device); body.train()
    if kind=='packet':
        source=torch.from_numpy(ext); decoder=nn.Linear(128,PACKET_BUDGET).to(device); body_fn=lambda x:body.packet_body(x.unsqueeze(1))
    else:
        source=torch.from_numpy(windows(ext)); decoder=nn.Linear(128,WINDOW_DIM).to(device); body_fn=lambda x:body.window_body(x)
    loader=DataLoader(TensorDataset(source),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed_value))
    opt=torch.optim.AdamW(list(body.parameters())+list(decoder.parameters()),lr=1e-3,weight_decay=1e-4); losses=[]
    for ep in range(epochs):
        total=0.
        for (x,) in loader:
            x=x.to(device); mask=(torch.rand_like(x)>0.15).float(); masked=x*mask
            z=body_fn(masked); loss=nn.functional.mse_loss(decoder(z),x); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(x)
        losses.append(total/len(source))
    return (body.packet_body.state_dict() if kind=='packet' else body.window_body.state_dict()), losses

def load_td():
    manifest=json.loads(PRIOR.read_text())['sampling']; data={}
    for role in ('source','valid','day14','day30','day90','day150','day270'):
        item=manifest[role]
        with np.load(DATA_ROOT/item['path'],allow_pickle=False) as with_npz:
            rows=with_npz['X'][item['rows']].astype(np.float32); y=with_npz['y'][item['rows']].astype(np.int64)
        data[role]={'p':np.sign(rows[:,:PACKET_BUDGET]).astype(np.float32),'w':windows(rows),'y':y}
    # Fixed five-shot per class, taking the first rows in the frozen manifest order.
    inds=[]
    for c in range(102): inds.extend(np.where(data['source']['y']==c)[0][:5])
    data['train_idx']=np.asarray(inds,dtype=np.int64)
    mean=data['source']['w'].mean(0); std=data['source']['w'].std(0); std[std<1e-6]=1
    for d in data:
        if isinstance(data[d],dict): data[d]['w']=(data[d]['w']-mean)/std
    return data

def predict(model,p,w,device):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(p),128): out.append(model(torch.from_numpy(p[i:i+128]).to(device),torch.from_numpy(w[i:i+128]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def train_condition(condition, device_name):
    config=json.loads((RUN/'config.json').read_text()); device=torch.device(device_name); torch.set_num_threads(4)
    ext=np.load(RUN/'artifacts/pcap_sequences.npz')['packets']; data=load_td(); histories={}; rows=[]; predictions={}
    for s in SEEDS:
        seed(s); model=Fusion().to(device); pre_loss=[]
        if condition=='pcap_packet_pretrain': model.packet_body.load_state_dict(pretrain_body('packet',ext,8,device,s)[0]);
        if condition=='pcap_window_pretrain': model.window_body.load_state_dict(pretrain_body('window',ext,8,device,s)[0]);
        train_idx=data['train_idx']; loader=DataLoader(TensorDataset(torch.from_numpy(data['source']['p'][train_idx]),torch.from_numpy(data['source']['w'][train_idx]),torch.from_numpy(data['source']['y'][train_idx])),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(s))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); hist=[]; best=-1; best_state=None; best_ep=0
        for ep in range(1,21):
            model.train(); total=0.; correct=0
            for p,w,y in loader:
                p,w,y=p.to(device),w.to(device),y.to(device); loss=nn.functional.cross_entropy(model(p,w),y); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(y); correct+=int((model(p,w).argmax(1)==y).sum())
            vp=predict(model,data['valid']['p'],data['valid']['w'],device); sc=metric(data['valid']['y'],vp); rec={'epoch':ep,'train_loss':total/len(train_idx),'train_accuracy':correct/len(train_idx),**sc}; hist.append(rec)
            if sc['macro_f1']>best: best=sc['macro_f1']; best_ep=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state); key=f'{condition}_{s}'; histories[key]={'history':hist,'best_epoch':best_ep,'pretrain_loss':pre_loss}
        torch.save({'state_dict':best_state,'condition':condition,'seed':s,'best_epoch':best_ep},RUN/'checkpoints'/f'{key}_v2.pt')
        for role in ('source','valid','day14','day30','day90','day150','day270'):
            pr=predict(model,data[role]['p'],data[role]['w'],device); predictions[f'{role}_{key}']=pr; rows.append({'condition':condition,'seed':s,'role':role,'best_epoch':best_ep,**metric(data[role]['y'],pr)})
        print(condition,s,'valid_f1',best,flush=True)
    (RUN/'artifacts'/('history_'+condition+'_v2.json')).write_text(json.dumps(histories,indent=2));
    with (RUN/'artifacts'/('metrics_'+condition+'_v2.csv')).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['condition','seed','role','best_epoch','accuracy','macro_f1']); wr.writeheader(); wr.writerows(rows)
    np.savez_compressed(RUN/'artifacts'/('predictions_'+condition+'_v2.npz'),**predictions)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--condition'); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args()
    prepare()
    if a.condition: train_condition(a.condition,a.device)
if __name__=='__main__': main()
