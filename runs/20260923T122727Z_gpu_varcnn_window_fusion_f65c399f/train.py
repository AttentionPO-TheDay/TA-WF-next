from __future__ import annotations
import csv, hashlib, json, random, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ta_wf_next.models import VarCNNDirection
from ta_wf_next.traffic_views import generate_views

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
ROLES=("source","valid","day14","day30","day90","day150","day270")
CONDITIONS=("varcnn_only","varcnn_constant","varcnn_window")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
    a=np.bincount(y*102+p,minlength=102**2).reshape(102,102); d=a.sum(0)+a.sum(1)
    return {"accuracy":float(np.mean(y==p)),"macro_f1":float(np.divide(2*np.diag(a),d,out=np.zeros(102),where=d!=0).mean())}
def extract(traces):
    packets=np.sign(traces[:,:5000]).astype(np.float32); full=np.zeros((len(traces),240),np.float32); neutral=np.zeros_like(full)
    for i,row in enumerate(traces):
        views=generate_views(row[:5000],input_kind="signed_timestamp",budget=5000,window_sizes=(50,250)); off=0
        for width,ws in views.direction_windows:
            for j,w in enumerate(ws):
                c=off+2*j; pair=(w.positive_fraction,w.transition_fraction); full[i,c:c+2]=pair; neutral[i,c:c+2]=(0.5,0.0) if w.partial else pair
            off+=2*((5000+width-1)//width)
    return packets,full,neutral
class Fusion(nn.Module):
    def __init__(self):
        super().__init__(); self.var=VarCNNDirection(102)
        self.fine=nn.Sequential(nn.Conv1d(2,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.coarse=nn.Sequential(nn.Conv1d(2,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(640,512),nn.ReLU(),nn.Dropout(.5)); self.mlp=nn.Linear(512,102)
    def forward(self,p,w):
        fine=w[:,:200].reshape(-1,100,2).transpose(1,2); coarse=w[:,200:].reshape(-1,20,2).transpose(1,2)
        f=self.var.forward_features(p[:,None,:]); t=torch.cat((self.fine(fine),self.coarse(coarse)),1)
        return self.mlp(self.head(torch.cat((f,t),1)))
def init(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def predict(model,cond,p,w,dev,bs):
    model.eval(); out=[]
    with torch.inference_mode():
        for st in range(0,len(p),bs):
            x=torch.from_numpy(p[st:st+bs]).to(dev); logits=model(x[:,None,:])[0] if cond=="varcnn_only" else model(x,torch.from_numpy(w[st:st+bs]).to(dev)); out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out).astype(np.int16)
def main():
    started=time.monotonic(); cfg=json.loads((RUN/"config.json").read_text()); assert cfg["status"]=="frozen" and torch.cuda.is_available(); assert not (RUN/"artifacts/predictions.npz").exists() and not list((RUN/"checkpoints").glob("*.pt")); torch.set_num_threads(4); torch.set_num_interop_threads(1); dev=torch.device("cuda:0")
    sampling_path=ROOT/"runs"/cfg["sampling_run"]/"artifacts/manifest.json"; sampling=json.loads(sampling_path.read_text())["sampling"]; root=Path(json.loads((ROOT/"configs/datasets.json").read_text())["data_root"]); data={}; hashes={}; source_full=None
    for role in ROLES:
        it=sampling[role]
        with np.load(root/it["path"],allow_pickle=False) as ar: idx=np.asarray(it["rows"],np.int64); y=ar["y"][idx].astype(np.int64); rows=ar["X"][idx]
        assert len(idx)==it["count"] and len(np.unique(y))==102 and len(np.unique(idx))==len(idx) and np.isfinite(rows).all(); p,full,w=extract(rows); data[role]={"p":p,"w":w,"y":y}; hashes[role]={hashlib.sha256(x.astype(np.int8).tobytes()).hexdigest() for x in p}; source_full=full if role=="source" else source_full; print("loaded",role,len(y),flush=True)
    assert not hashes["source"]&hashes["valid"]; mean=source_full.mean(0); std=source_full.std(0); std[std<1e-6]=1.; np.savez_compressed(RUN/"artifacts/window_statistics.npz",mean=mean,std=std)
    for r in ROLES: data[r]["w"]=((data[r]["w"]-mean)/std).astype(np.float32)
    sealed=(RUN/"PLAN.md",RUN/"config.json",Path(__file__),sampling_path,ROOT/"configs/datasets.json",ROOT/"src/ta_wf_next/models/varcnn.py",ROOT/"src/ta_wf_next/traffic_views.py",RUN/"artifacts/window_statistics.npz"); (RUN/"artifacts/input_seal.json").write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sealed},indent=2))
    histories={}; preds={}; infos={}; mrows=[]
    for seed in cfg["training_seeds"]:
      basehash=None; fhash=None
      for cond in CONDITIONS:
        init(seed); model=(VarCNNDirection(102) if cond=="varcnn_only" else Fusion()).to(dev); base=model if cond=="varcnn_only" else model.var; h=hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in base.parameters())).hexdigest(); basehash=h if basehash is None else basehash; assert h==basehash
        if cond!="varcnn_only":
          q=hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in model.parameters())).hexdigest(); fhash=q if fhash is None else fhash; assert q==fhash
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]); src=data["source"]; tw=np.zeros_like(src["w"]) if cond=="varcnn_constant" else src["w"]; loader=DataLoader(TensorDataset(torch.from_numpy(src["p"]),torch.from_numpy(tw),torch.from_numpy(src["y"])),batch_size=cfg["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed)); hist=[]; best=-1.; began=time.monotonic()
        for ep in range(1,46):
          model.train(); loss_sum=0.; correct=0
          for p,w,y in loader:
            p,w,y=p.to(dev),w.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); logits=model(p[:,None,:])[0] if cond=="varcnn_only" else model(p,w); loss=nn.functional.cross_entropy(logits,y); assert torch.isfinite(loss); loss.backward(); opt.step(); loss_sum+=float(loss)*len(y); correct+=int((logits.argmax(1)==y).sum())
          va=data["valid"]; vw=np.zeros_like(va["w"]) if cond=="varcnn_constant" else va["w"]; sc=metric(va["y"],predict(model,cond,va["p"],vw,dev,64)); hist.append({"epoch":ep,"train_loss":loss_sum/len(src["y"]),"train_accuracy":correct/len(src["y"]),**sc})
          if sc["macro_f1"]>best: best=sc["macro_f1"]; beste=ep; state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
          if time.monotonic()-started>cfg["time_limit_seconds"]: raise TimeoutError("GPU budget exceeded; partial run retained")
        key=f"{cond}_{seed}"; model.load_state_dict(state); histories[key]=hist; infos[key]={"best_epoch":beste,"valid_macro_f1":best,"parameters":sum(x.numel() for x in model.parameters()),"initial_base_hash":basehash,"initial_fusion_hash":fhash,"train_seconds":time.monotonic()-began}; torch.save({"state_dict":state,"epoch":beste,"condition":cond,"seed":seed},RUN/"checkpoints"/f"{key}.pt")
        for role in ROLES:
          z=data[role]; zw=np.zeros_like(z["w"]) if cond=="varcnn_constant" else z["w"]; pr=predict(model,cond,z["p"],zw,dev,64); preds[f"{role}_{key}"]=pr; mrows.append({"role":role,"condition":cond,"seed":seed,"best_epoch":beste,**metric(z["y"],pr)})
        print(key,"valid",best,"epoch",beste,flush=True)
    (RUN/"artifacts/history.json").write_text(json.dumps(histories,indent=2));
    with (RUN/"artifacts/metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=("role","condition","seed","best_epoch","accuracy","macro_f1")); w.writeheader(); w.writerows(mrows)
    np.savez_compressed(RUN/"artifacts/predictions.npz",**preds); (RUN/"artifacts/manifest.json").write_text(json.dumps({"sampling_manifest":str(sampling_path.relative_to(ROOT)),"sampling_sha256":sha(sampling_path),"prediction_sha256":sha(RUN/"artifacts/predictions.npz"),"models":infos,"elapsed_seconds":time.monotonic()-started},indent=2)); print("COMPLETE")
if __name__=="__main__": main()
