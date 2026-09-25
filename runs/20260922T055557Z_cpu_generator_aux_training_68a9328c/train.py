import os, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ta_wf_next.traffic_views import generate_views
R=Path(__file__).resolve().parent; ROOT=R.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p):
    cm=np.bincount(y*102+p,minlength=10404).reshape(102,102); t=cm.diagonal(); den=cm.sum(0)+cm.sum(1)
    return dict(accuracy=float((y==p).mean()),macro_f1=float(np.divide(2*t,den,out=np.zeros(102),where=den!=0).mean()))
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(1,16,9,stride=4),nn.ReLU(),nn.Conv1d(16,32,7,stride=4),nn.ReLU(),nn.AdaptiveAvgPool1d(32))
        self.head=nn.Linear(1024,102)
    def features(self,x): return self.body(x[:,None,:]).flatten(1)
    def forward(self,x): return self.head(self.features(x))
def targets(rows):
    w=np.zeros((len(rows),240),np.float32); r=np.zeros((len(rows),8),np.float32)
    for i,row in enumerate(rows):
        v=generate_views(row,input_kind='direction',budget=5000)
        offset=0
        for width,entries in v.direction_windows:
            for j,t in enumerate(entries): w[i,offset+2*j:offset+2*j+2]=[t.positive_fraction,t.transition_fraction]
            offset+=2*((5000+width-1)//width)
        for k,sign in enumerate([1,-1]):
            a=np.array([t.count for t in v.runs.runs if t.direction==sign])
            if len(a): r[i,4*k:4*k+4]=np.log1p([len(a),a.mean(),a.std(),a.max()])
    return w,r
def standardize(a):
    mean=a.mean(0); std=a.std(0); std=np.where(std<1e-6,1.,std)
    return (a-mean)/std,mean,std
def main():
    c=json.loads((R/'config.json').read_text()); assert c['status']=='frozen' and not c['gpu']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not list((R/'checkpoints').glob('*.pt'))
    torch.set_num_threads(c['threads']); torch.set_num_interop_threads(1)
    data=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
    old=ROOT/'runs'/c['manifest_source']/'artifacts/manifest.json'
    manifest=json.loads(old.read_text())['sampling']; arrays={}; ys={}; hashes={}
    for role in c['roles']:
        item=manifest[role]
        with np.load(data/item['path'],allow_pickle=False) as z:
            arrays[role]=np.sign(z['X'][item['rows'],:5000]).astype(np.float32); ys[role]=z['y'][item['rows']].astype(np.int64)
        assert np.isfinite(arrays[role]).all() and set(ys[role])==set(range(102))
        hashes[role]=[hashlib.sha256(row.astype(np.int8).tobytes()).hexdigest() for row in arrays[role]]
    assert not set(hashes['source'])&set(hashes['valid'])
    w,r=targets(arrays['source']); x,mean,std=standardize(arrays['source'])
    vw=(arrays['valid']-mean)/std
    w,wm,ws=standardize(w); r,rm,rs=standardize(r)
    np.savez_compressed(R/'artifacts/statistics.npz',packet_mean=mean,packet_std=std,window_mean=wm,window_std=ws,run_mean=rm,run_std=rs)
    np.savez_compressed(R/'artifacts/targets.npz',windows=w,runs=r)
    (R/'artifacts/manifest.json').write_text(json.dumps(dict(sampling={k:manifest[k] for k in c['roles']},direction_hashes=hashes,source_manifest_sha256=sha(old)),indent=2))
    seal_files=[R/'PLAN.md',R/'config.json',Path(__file__),ROOT/'src/ta_wf_next/traffic_views.py',ROOT/'src/ta_wf_next/burst_tokens.py']
    (R/'artifacts/pretraining_seal.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in seal_files},indent=2))
    x=torch.from_numpy(x); vx=torch.from_numpy(vw); y=torch.from_numpy(ys['source']); w=torch.from_numpy(w); r=torch.from_numpy(r)
    histories={}; preds={}; info={}
    for seed in c['seeds']:
      init_hash=None
      for condition in c['conditions']:
        key=f'{condition}_{seed}'; torch.manual_seed(seed); model=Model()
        h=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        if init_hash is None: init_hash=h
        assert init_hash==h
        wh=nn.Linear(1024,240); rh=nn.Linear(1024,8)
        opt=torch.optim.AdamW(list(model.parameters())+list(wh.parameters())+list(rh.parameters()),lr=c['lr'],weight_decay=c['weight_decay'])
        gen=torch.Generator().manual_seed(seed); hist=[]; best=-1.; start=time.perf_counter()
        for epoch in range(1,c['epochs']+1):
            model.train(); ce_sum=aux_sum=correct=0
            order=torch.randperm(len(x),generator=gen)
            for ids in order.split(c['batch_size']):
                opt.zero_grad(set_to_none=True); f=model.features(x[ids]); logits=model.head(f)
                ce=nn.functional.cross_entropy(logits,y[ids])
                if condition=='window_aux': aux=nn.functional.mse_loss(wh(f),w[ids])
                elif condition=='run_aux': aux=nn.functional.mse_loss(rh(f),r[ids])
                else: aux=ce.new_zeros(())
                loss=ce+c['aux_weight']*aux; assert torch.isfinite(loss)
                loss.backward(); opt.step()
                ce_sum+=ce.item()*len(ids); aux_sum+=aux.item()*len(ids); correct+=int((logits.argmax(1)==y[ids]).sum())
            model.eval()
            with torch.inference_mode(): p=torch.cat([model(b).argmax(1) for b in vx.split(c['batch_size'])]).numpy()
            m=metrics(ys['valid'],p); hist.append(dict(epoch=epoch,ce=ce_sum/len(x),aux_mse=aux_sum/len(x),train_accuracy=correct/len(x),**m))
            if m['macro_f1']>best:
                best=m['macro_f1']; state={k:v.clone() for k,v in model.state_dict().items()}; best_epoch=epoch
                aux_state={'window':wh.state_dict(),'run':rh.state_dict()}
                aux_state={k:{a:b.clone() for a,b in v.items()} for k,v in aux_state.items()}
        model.load_state_dict(state)
        with torch.inference_mode(): p=torch.cat([model(b).argmax(1) for b in vx.split(c['batch_size'])]).numpy()
        torch.save(state,R/'checkpoints'/f'{key}.pt'); torch.save(aux_state,R/'checkpoints'/f'{key}_aux.pt')
        reload=Model(); reload.load_state_dict(torch.load(R/'checkpoints'/f'{key}.pt',weights_only=True)); reload.eval()
        with torch.inference_mode(): torch.testing.assert_close(model(vx[:8]),reload(vx[:8]),rtol=0,atol=0)
        preds[key]=p; histories[key]=hist
        info[key]=dict(best_epoch=best_epoch,seconds=time.perf_counter()-start,initial_hash=h,inference_parameters=sum(p.numel() for p in model.parameters()))
        (R/'artifacts/history_partial.json').write_text(json.dumps(histories,indent=2))
        print(key,info[key],flush=True)
    np.savez_compressed(R/'artifacts/predictions.npz',**preds)
    (R/'artifacts/history.json').write_text(json.dumps(histories,indent=2))
    (R/'artifacts/models.json').write_text(json.dumps(info,indent=2))
    files=list((R/'checkpoints').glob('*.pt'))+[R/'artifacts/predictions.npz',R/'artifacts/statistics.npz',R/'artifacts/targets.npz']
    (R/'artifacts/output_seal.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
    rows=[dict(condition=k,**metrics(ys['valid'],p),**info[k]) for k,p in preds.items()]
    (R/'artifacts/metrics.json').write_text(json.dumps(rows,indent=2)); print('COMPLETE',flush=True)
if __name__=='__main__': main()
